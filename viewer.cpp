// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

#include "application.hpp"
#include "command_buffer.hpp"
#include "device.hpp"
#include "os_filesystem.hpp"
#include "muglm/muglm_impl.hpp"
#include "pyrowave_encoder.hpp"
#include "pyrowave_decoder.hpp"
#include "yuv4mpeg.hpp"
#include "pyrowave_common.hpp"
#include "flat_renderer.hpp"
#include "ui_manager.hpp"
#include <string.h>
#include <stdexcept>

using namespace Granite;
using namespace Vulkan;

struct YCbCrImages
{
	Vulkan::ImageHandle images[3];
	PyroWave::ViewBuffers views;
};

static YCbCrImages create_ycbcr_images(Device &device, int width, int height, VkFormat fmt, PyroWave::ChromaSubsampling chroma)
{
	YCbCrImages images;
	auto info = ImageCreateInfo::immutable_2d_image(width, height, fmt);
	info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
	             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;

	images.images[0] = device.create_image(info);
	device.set_name(*images.images[0], "Y");

	if (chroma == PyroWave::ChromaSubsampling::Chroma420)
	{
		info.width >>= 1;
		info.height >>= 1;
	}

	images.images[1] = device.create_image(info);
	device.set_name(*images.images[1], "Cb");

	images.images[2] = device.create_image(info);
	device.set_name(*images.images[2], "Cr");

	for (int i = 0; i < 3; i++)
		images.views.planes[i] = &images.images[i]->get_view();

	return images;
}

struct ViewerApplication : Granite::Application, Granite::EventHandler
{
	explicit ViewerApplication(const char *path_)
		: path(path_)
	{
		if (!file.open_read(path))
			throw std::runtime_error("Failed to open.");

		get_wsi().set_backbuffer_format(BackbufferFormat::UNORM);
		EVENT_MANAGER_REGISTER_LATCH(ViewerApplication, on_device_created, on_device_destroyed, DeviceCreatedEvent);
		EVENT_MANAGER_REGISTER(ViewerApplication, on_key_press, KeyboardEvent);
		EVENT_MANAGER_REGISTER(ViewerApplication, on_mouse, MouseMoveEvent);
		EVENT_MANAGER_REGISTER(ViewerApplication, on_mouse_event, MouseButtonEvent);

		x_slide = file.get_width() / 2;
	}

	bool is_mouse_active = false;
	bool paused = false;

	enum class Mode
	{
		Slide,
		Flicker,
		Delta
	};
	Mode mode = Mode::Slide;

	bool on_mouse(const MouseMoveEvent &e)
	{
		if (is_mouse_active)
			x_slide = int(e.get_abs_x());
		return true;
	}

	bool on_mouse_event(const MouseButtonEvent &e)
	{
		is_mouse_active = e.get_pressed();
		return true;
	}

	bool on_key_press(const KeyboardEvent &e)
	{
		if (e.get_key_state() != KeyState::Released)
		{
			if (e.get_key() == Key::Up)
				bit_rate_mbit += 10;
			else if (e.get_key() == Key::Down && bit_rate_mbit > 20)
				bit_rate_mbit -= 10;
			else if (e.get_key() == Key::F)
				mode = Mode::Flicker;
			else if (e.get_key() == Key::D)
				mode = Mode::Delta;
			else if (e.get_key() == Key::S)
				mode = Mode::Slide;
			else if (e.get_key() == Key::P)
			{
				get_wsi().set_backbuffer_format(
						get_wsi().get_backbuffer_format() == BackbufferFormat::HDR10 ?
						BackbufferFormat::UNORM : BackbufferFormat::HDR10);
			}
		}

		if (e.get_key_state() == KeyState::Pressed && e.get_key() == Key::Space)
			paused = !paused;

		return true;
	}

	void on_device_created(const DeviceCreatedEvent &e)
	{
		auto format = YUV4MPEGFile::format_to_bytes_per_component(file.get_format()) == 2 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R8_UNORM;
		auto chroma = YUV4MPEGFile::format_has_subsampling(file.get_format()) ? PyroWave::ChromaSubsampling::Chroma420 : PyroWave::ChromaSubsampling::Chroma444;
		in_images = create_ycbcr_images(e.get_device(), file.get_width(), file.get_height(), format, chroma);
		out_images = create_ycbcr_images(e.get_device(), file.get_width(), file.get_height(), format, chroma);
		enc.init(&e.get_device(), file.get_width(), file.get_height(), chroma);
		dec.init(&e.get_device(), file.get_width(), file.get_height(), chroma);
	}

	void on_device_destroyed(const DeviceCreatedEvent &)
	{
		in_images = {};
		out_images = {};
	}

	void render_frame(double, double elapsed_time) override
	{
		auto &device = get_wsi().get_device();
		auto cmd = device.request_command_buffer();

		if (!paused)
		{
			if (!file.begin_frame())
			{
				file = {};
				if (!file.open_read(path) || !file.begin_frame())
				{
					request_shutdown();
					return;
				}
			}

			for (int i = 0; i < 3; i++)
			{
				cmd->image_barrier(*in_images.images[i], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				                   0, 0,
				                   VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
			}

			for (int i = 0; i < 3; i++)
			{
				auto *y = cmd->update_image(*in_images.images[i]);
				if (!file.read(y, in_images.images[i]->get_width() * in_images.images[i]->get_height() * YUV4MPEGFile::format_to_bytes_per_component(file.get_format())))
				{
					LOGE("Failed to read plane.\n");
					device.submit_discard(cmd);
					request_shutdown();
					return;
				}
			}

			for (int i = 0; i < 3; i++)
			{
				cmd->image_barrier(*in_images.images[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				                   VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
				                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
			}
		}

		unsigned bitstream_size = bit_rate_mbit * 1000000ull / (60 * 8);
		bitstream_size &= ~3u;

		BufferCreateInfo buffer_info = {};
		buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

		buffer_info.size = enc.get_meta_required_size();
		buffer_info.domain = BufferDomain::Device;
		auto meta = device.create_buffer(buffer_info);
		buffer_info.domain = BufferDomain::CachedHost;
		auto meta_host = device.create_buffer(buffer_info);

		buffer_info.size = bitstream_size + 2 * enc.get_meta_required_size();
		buffer_info.domain = BufferDomain::Device;
		auto bitstream = device.create_buffer(buffer_info);
		buffer_info.domain = BufferDomain::CachedHost;
		auto bitstream_host = device.create_buffer(buffer_info);

		PyroWave::Encoder::BitstreamBuffers buffers = {};
		buffers.meta.buffer = meta.get();
		buffers.meta.size = meta->get_create_info().size;
		buffers.bitstream.buffer = bitstream.get();
		buffers.bitstream.size = bitstream->get_create_info().size;
		buffers.target_size = bitstream_size;

		enc.encode(*cmd, in_images.views, buffers);
		cmd->copy_buffer(*bitstream_host, *bitstream);
		cmd->copy_buffer(*meta_host, *meta);
		cmd->barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
		             VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);

		Fence fence;
		device.submit(cmd, &fence);
		fence->wait();

		auto *mapped_meta = static_cast<const PyroWave::BitstreamPacket *>(
				device.map_host_buffer(*meta_host, MEMORY_ACCESS_READ_BIT));
		auto *mapped_bits = static_cast<const uint32_t *>(
				device.map_host_buffer(*bitstream_host, MEMORY_ACCESS_READ_BIT));

		std::vector<uint8_t> reordered_packet_buffer(bitstream_size * 2);
		size_t num_packets = enc.compute_num_packets(mapped_meta, 8 * 1024);
		std::vector<PyroWave::Encoder::Packet> packets(num_packets);
		size_t out_packets = enc.packetize(packets.data(), 8 * 1024,
		                                   reordered_packet_buffer.data(),
		                                   reordered_packet_buffer.size(),
		                                   mapped_meta, mapped_bits);
		(void)out_packets;

		size_t encoded_size = 0;
		for (auto &p : packets)
			encoded_size += p.size;

		LOGI("Total encoded size: %zu\n", encoded_size);

		if (encoded_size > bitstream_size)
		{
			LOGE("Broken rate control\n");
			return;
		}

		assert(out_packets == num_packets);

		for (auto &p : packets)
			if (!dec.push_packet(reordered_packet_buffer.data() + p.offset, p.size))
				return;

		cmd = device.request_command_buffer();

		for (int i = 0; i < 3; i++)
		{
			cmd->image_barrier(*out_images.images[i], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
			                   0, 0,
			                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
		}

		dec.decode(*cmd, out_images.views);

		for (int i = 0; i < 3; i++)
		{
			cmd->image_barrier(*out_images.images[i], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
			                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
		}

		cmd->begin_render_pass(device.get_swapchain_render_pass(SwapchainRenderPass::ColorOnly));
		cmd->set_sampler(0, 3, StockSampler::LinearClamp);

		cmd->set_specialization_constant_mask(3);
		cmd->set_specialization_constant(0, file.get_format() == YUV4MPEGFile::Format::YUV420P16);
		cmd->set_specialization_constant(1, file.is_full_range());

		CommandBufferUtil::setup_fullscreen_quad(*cmd, "builtin://shaders/quad.vert", "assets://yuv2rgb.frag",
		                                         {{ "DELTA", mode == Mode::Delta ? 1 : 0 }});

		x_slide = clamp(x_slide, 50, int(cmd->get_viewport().width) - 50);

		const float full_color = get_wsi().get_backbuffer_format() == BackbufferFormat::HDR10 ? 0.75f : 1.0f;

		if (mode == Mode::Flicker)
		{
			if (muglm::fract(elapsed_time * 10.0) < 0.5)
			{
				cmd->set_texture(0, 0, *in_images.views.planes[0]);
				cmd->set_texture(0, 1, *in_images.views.planes[1]);
				cmd->set_texture(0, 2, *in_images.views.planes[2]);
			}
			else
			{
				cmd->set_texture(0, 0, *out_images.views.planes[0]);
				cmd->set_texture(0, 1, *out_images.views.planes[1]);
				cmd->set_texture(0, 2, *out_images.views.planes[2]);
			}

			cmd->draw(3);
			flat_renderer.begin();
			char text[64];
			snprintf(text, sizeof(text), "FLICKER %u mbits | %.3f bpp @ 60 fps%s",
			         bit_rate_mbit,
					 double(bitstream_size * 8) / double(file.get_width() * file.get_height()),
			         paused ? " (paused)" : "");
			flat_renderer.render_text(GRANITE_UI_MANAGER()->get_font(UI::FontSize::Large),
			                          text, vec3(20, 20, 0), vec2(400, 200), vec4(full_color, full_color, 0.0f, 1.0f),
			                          Font::Alignment::TopLeft);
			flat_renderer.flush(*cmd, vec3(0), vec3(cmd->get_viewport().width, cmd->get_viewport().height, 1));
		}
		else if (mode == Mode::Slide)
		{
			cmd->set_texture(0, 0, *in_images.views.planes[0]);
			cmd->set_texture(0, 1, *in_images.views.planes[1]);
			cmd->set_texture(0, 2, *in_images.views.planes[2]);
			cmd->set_scissor({{ 0, 0 }, { uint32_t(x_slide), uint32_t(cmd->get_viewport().height) }});
			cmd->draw(3);

			cmd->set_texture(0, 0, *out_images.views.planes[0]);
			cmd->set_texture(0, 1, *out_images.views.planes[1]);
			cmd->set_texture(0, 2, *out_images.views.planes[2]);
			cmd->set_scissor({{ int32_t(x_slide), 0 },
			                  { uint32_t(cmd->get_viewport().width), uint32_t(cmd->get_viewport().height) }});
			cmd->draw(3);

			cmd->set_scissor({{ 0, 0 },
			                  { uint32_t(cmd->get_viewport().width), uint32_t(cmd->get_viewport().height) }});

			flat_renderer.begin();
			char text[64];
			snprintf(text, sizeof(text), "%u mbits | %.3f bpp @ 60 fps%s", bit_rate_mbit,
			         double(bitstream_size * 8) / double(file.get_width() * file.get_height()),
			         paused ? " (paused)" : "");
			flat_renderer.render_text(GRANITE_UI_MANAGER()->get_font(UI::FontSize::Large),
			                          text, vec3(20, 20, 0), vec2(400, 200), vec4(full_color, full_color, 0.0f, 1.0f),
			                          Font::Alignment::TopLeft);
			flat_renderer.render_text(GRANITE_UI_MANAGER()->get_font(UI::FontSize::Large),
			                          text, vec3(18, 22, 0.5f), vec2(400, 200), vec4(0.0f, 0.0f, 0.0f, 1.0f),
			                          Font::Alignment::TopLeft);
			flat_renderer.render_quad(vec3(float(x_slide), 0.0f, 0.8f),
			                          vec2(2.0f, cmd->get_viewport().height),
			                          vec4(full_color, full_color, 0.0f, 1.0f));
			flat_renderer.flush(*cmd, vec3(0), vec3(cmd->get_viewport().width, cmd->get_viewport().height, 1));
		}
		else
		{
			cmd->set_texture(0, 0, *in_images.views.planes[0]);
			cmd->set_texture(0, 1, *out_images.views.planes[0]);
			cmd->draw(3);

			flat_renderer.begin();
			char text[64];
			snprintf(text, sizeof(text), "DELTA %u mbits | %.3f bpp @ 60 fps%s",
			         bit_rate_mbit,
			         double(bitstream_size * 8) / double(file.get_width() * file.get_height()),
			         paused ? " (paused)" : "");
			flat_renderer.render_text(GRANITE_UI_MANAGER()->get_font(UI::FontSize::Large),
			                          text, vec3(20, 20, 0), vec2(400, 200), vec4(full_color, full_color, 0.0f, 1.0f),
			                          Font::Alignment::TopLeft);
			flat_renderer.flush(*cmd, vec3(0), vec3(cmd->get_viewport().width, cmd->get_viewport().height, 1));
		}

		cmd->end_render_pass();

		device.submit(cmd);
	}

	unsigned get_default_width() override
	{
		return file.get_width();
	}

	unsigned get_default_height() override
	{
		return file.get_height();
	}

	PyroWave::Encoder enc;
	PyroWave::Decoder dec;
	YCbCrImages in_images;
	YCbCrImages out_images;
	YUV4MPEGFile file;
	const char *path;
	unsigned bit_rate_mbit = 200;
	FlatRenderer flat_renderer;

	int x_slide = 100;
};

namespace Granite
{
Application *application_create(int argc, char **argv)
{
	GRANITE_APPLICATION_SETUP_FILESYSTEM();

	if (argc != 2)
	{
		LOGE("Usage: pyrowave-viewer test.y4m\n");
		return nullptr;
	}

	try
	{
		auto *app = new ViewerApplication(argv[1]);
		return app;
	}
	catch (const std::exception &e)
	{
		LOGE("application_create() threw exception: %s\n", e.what());
		return nullptr;
	}
}
}
