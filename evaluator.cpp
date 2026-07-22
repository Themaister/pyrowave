// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

#include <random>

#include "application.hpp"
#include "command_buffer.hpp"
#include "device.hpp"
#include "os_filesystem.hpp"
#include "muglm/muglm_impl.hpp"
#include "yuv4mpeg.hpp"
#include "pyrowave_common.hpp"
#include "flat_renderer.hpp"
#include "ui_manager.hpp"
#include "pyrowave_decoder.hpp"
#include "pyrowave_encoder.hpp"
#include <string.h>
#include <stdexcept>
#include "rapidjson_wrapper.hpp"
#include "path_utils.hpp"

using namespace Granite;
using namespace Vulkan;

struct YCbCrImages
{
	ImageHandle images[3];
};

enum class Codec
{
	None,
	PyroWave
};

struct TestClip
{
	std::unique_ptr<YUV4MPEGFile> file;
	Codec codec = Codec::None;
	int codec_mbits = 0;
	std::string name;
	std::string desc;
};

struct TestClipGroup
{
	std::string name;
	std::vector<TestClip> clips;
};

// BT.500 DSIS.
enum SubIterations
{
	FirstReferenceSequence = 0,
	MidGray0 = 1,
	FirstTestSequence = 2,
	MidGray1 = 3,
	SecondReferenceSequence = 4,
	MidGray2 = 5,
	SecondTestSequence = 6,
	MidGrayVote = 7,
	SubIterationCount
};

static std::vector<TestClipGroup> parse_test_clips(const std::string &path)
{
	using namespace rapidjson;
	std::vector<TestClipGroup> parsed_clips;

	std::string str;
	if (!GRANITE_FILESYSTEM()->read_file_to_string(path, str))
		throw std::runtime_error("Failed to parse test clips.");

	Document doc;
	doc.Parse(str);
	if (doc.HasParseError())
		throw std::runtime_error("Failed to parse.");

	auto &tests = doc["tests"];

	for (auto test_itr = tests.MemberBegin(); test_itr != tests.MemberEnd(); ++test_itr)
	{
		TestClipGroup clip_group;
		clip_group.name = test_itr->name.GetString();
		auto &clips = test_itr->value;

		for (auto itr = clips.Begin(); itr != clips.End(); ++itr)
		{
			auto &clip = *itr;
			TestClip parsed_clip;

			if (clip.HasMember("codec"))
			{
				if (itr == clips.Begin())
					throw std::logic_error("First clip cannot be a codec derived input.");

				if (strcmp(clip["codec"].GetString(), "pyrowave") == 0)
				{
					parsed_clip.codec = Codec::PyroWave;
					parsed_clip.codec_mbits = clip["mbits"].GetInt();
				}
			}

			if (parsed_clip.codec == Codec::None)
			{
				parsed_clip.file = std::make_unique<YUV4MPEGFile>();
				auto clip_path = Path::relpath(path, clip["path"].GetString());
				if (!parsed_clip.file->open_read(clip_path))
				{
					LOGE("Failed to open %s for reading.\n", clip_path.c_str());
					throw std::runtime_error("Failed to parse.");
				}
			}

			parsed_clip.name = clip["name"].GetString();
			parsed_clip.desc = clip["desc"].GetString();
			clip_group.clips.push_back(std::move(parsed_clip));
		}

		parsed_clips.push_back(std::move(clip_group));
	}

	return parsed_clips;
}

static YCbCrImages create_ycbcr_images(Device &device, int width, int height, VkFormat fmt, PyroWave::ChromaSubsampling chroma)
{
	YCbCrImages images;
	auto info = ImageCreateInfo::immutable_2d_image(width, height, fmt);
	info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
	             VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
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

	return images;
}

struct EvaluatorApplication : Application, EventHandler
{
	EvaluatorApplication(const char *path, const char *csv)
	{
		test_clips = parse_test_clips(path);
		if (test_clips.size() < 2)
			throw std::runtime_error("Need at least two test clips.");

		evaluation_file.reset(fopen(csv, "w"));
		if (!evaluation_file)
			throw std::runtime_error("Failed to open output CSV.");
		fprintf(evaluation_file.get(), "clip,test,vote\n");

		random_engine.seed(Util::get_current_time_nsecs());

		get_wsi().set_backbuffer_format(BackbufferFormat::UNORM);
		EVENT_MANAGER_REGISTER_LATCH(EvaluatorApplication, on_device_created, on_device_destroyed, DeviceCreatedEvent);
		EVENT_MANAGER_REGISTER(EvaluatorApplication, on_key_press, KeyboardEvent);
	}

	void register_voting(int score)
	{
		if (current_sub_iteration >= SecondReferenceSequence)
			current_voting = score;
	}

	bool on_key_press(const KeyboardEvent &e)
	{
		if (e.get_key_state() == KeyState::Pressed)
		{
			if (e.get_key() == Key::_1)
				register_voting(1);
			else if (e.get_key() == Key::_2)
				register_voting(2);
			else if (e.get_key() == Key::_3)
				register_voting(3);
			else if (e.get_key() == Key::_4)
				register_voting(4);
			else if (e.get_key() == Key::_5)
				register_voting(5);
			else if (e.get_key() == Key::_0)
				register_voting(0);
			else if (e.get_key() == Key::V)
				debug_enable = !debug_enable;
		}

		return true;
	}

	void on_device_created(const DeviceCreatedEvent &e)
	{
		auto &representative_file = *test_clips.front().clips.front().file;

		for (auto &clip_group : test_clips)
		{
			for (auto &clip : clip_group.clips)
			{
				if (!clip.file)
					continue;

				auto divided_fps_num = int64_t(clip.file->get_frame_rate_num()) * representative_file.get_frame_rate_den();
				auto divided_fps_den = int64_t(clip.file->get_frame_rate_den()) * representative_file.get_frame_rate_num();

				if (clip.file->get_format() != representative_file.get_format() ||
					clip.file->get_width() != representative_file.get_width() ||
					clip.file->get_height() != representative_file.get_height() ||
					clip.file->is_full_range() != representative_file.is_full_range() ||
					divided_fps_den != divided_fps_num)
				{
					throw std::runtime_error("Mismatch in clip parameters.");
				}
			}
		}

		auto format = YUV4MPEGFile::format_to_bytes_per_component(
			              representative_file.get_format()) == 2
			              ? VK_FORMAT_R16_UNORM
			              : VK_FORMAT_R8_UNORM;

		auto chroma = YUV4MPEGFile::format_has_subsampling(representative_file.get_format())
					  ? PyroWave::ChromaSubsampling::Chroma420 : PyroWave::ChromaSubsampling::Chroma444;

		images = create_ycbcr_images(e.get_device(), representative_file.get_width(), representative_file.get_height(), format, chroma);
		get_wsi().set_enable_timing_feedback(true);

		encoder.init(&e.get_device(), representative_file.get_width(), representative_file.get_height(), chroma);
		decoder.init(&e.get_device(), representative_file.get_width(), representative_file.get_height(), chroma);
	}

	void on_device_destroyed(const DeviceCreatedEvent &)
	{
		images = {};
	}

	PyroWave::Encoder encoder;
	PyroWave::Decoder decoder;
	int current_iteration = -1;
	int current_sub_iteration = 0;
	int current_clip_index = 0;
	int current_test_index = 0;
	int current_voting = 0;
	double iteration_start_time = 0.0;
	double current_sub_iteration_time = 0.0;
	bool debug_enable = false;
	std::default_random_engine random_engine;

	void roundtrip_pyrowave(CommandBufferHandle &cmd, int mbits)
	{
		auto &representative_file = *test_clips.front().clips.front().file;

		int bits_per_frame = int(1000000ll * mbits * representative_file.get_frame_rate_den() / representative_file.get_frame_rate_num());
		int bytes_per_frame = bits_per_frame / 8;
		bytes_per_frame &= ~3;

		auto &device = cmd->get_device();

		PyroWave::ViewBuffers views = {};
		for (int i = 0; i < 3; i++)
			views.planes[i] = &images.images[i]->get_view();

		BufferCreateInfo bufinfo = {};
		bufinfo.size = bytes_per_frame + encoder.get_meta_required_size();
		bufinfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		bufinfo.domain = BufferDomain::Device;
		auto bitstream_gpu = device.create_buffer(bufinfo);
		bufinfo.domain = BufferDomain::CachedHost;
		auto bitstream_cpu = device.create_buffer(bufinfo);

		bufinfo.size = encoder.get_meta_required_size();
		bufinfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		bufinfo.domain = BufferDomain::Device;
		auto meta_gpu = device.create_buffer(bufinfo);
		bufinfo.domain = BufferDomain::CachedHost;
		auto meta_cpu = device.create_buffer(bufinfo);

		PyroWave::Encoder::BitstreamBuffers buffers = {};
		buffers.target_size = bytes_per_frame;
		buffers.bitstream.buffer = bitstream_gpu.get();
		buffers.bitstream.size = bitstream_gpu->get_create_info().size;
		buffers.meta.buffer = meta_gpu.get();
		buffers.meta.size = meta_gpu->get_create_info().size;

		encoder.encode(*cmd, views, buffers);

		cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
		             VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
		cmd->copy_buffer(*bitstream_cpu, *bitstream_gpu);
		cmd->copy_buffer(*meta_cpu, *meta_gpu);
		cmd->barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
		             VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
		Fence fence;
		device.submit(cmd, &fence);
		fence->wait();

		std::unique_ptr<uint8_t[]> bitstream(new uint8_t[bytes_per_frame]);
		auto *mapped_bitstream = device.map_host_buffer(*bitstream_cpu, MEMORY_ACCESS_READ_BIT);
		auto *mapped_meta = device.map_host_buffer(*meta_cpu, MEMORY_ACCESS_READ_BIT);
		PyroWave::Encoder::Packet packet = {};
		encoder.packetize(&packet, bytes_per_frame, bitstream.get(), bytes_per_frame,
		                  mapped_meta, mapped_bitstream);

		cmd = device.request_command_buffer();

		for (auto &img : images.images)
		{
			cmd->image_barrier(*img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
			                   VK_PIPELINE_STAGE_NONE, VK_ACCESS_NONE,
			                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
		}

		decoder.clear();
		decoder.push_packet(bitstream.get() + packet.offset, packet.size);
		decoder.decode(*cmd, views);

		for (auto &img : images.images)
		{
			cmd->image_barrier(*img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
			                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
			                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
		}
	}

	void iterate(CommandBufferHandle &cmd, double elapsed_time)
	{
		static constexpr double SubIterationTimes[] = {
			10.0, 3.0, 10.0, 3.0,
			10.0, 3.0, 10.0, 8.0,
		};

		const auto time_to_sub_iteration = [](double &t) -> int
		{
			int i;
			for (i = 0; i < SubIterationCount && t >= SubIterationTimes[i]; i++)
				t -= SubIterationTimes[i];
			return i;
		};

		current_sub_iteration_time = elapsed_time - iteration_start_time;
		int sub_iteration_index = time_to_sub_iteration(current_sub_iteration_time);
		bool redraw = false;

		if  (sub_iteration_index == SubIterationCount)
		{
			fprintf(evaluation_file.get(), "%d,%d,%d\n",
			        current_clip_index, current_test_index, current_voting);
		}

		if (sub_iteration_index == SubIterationCount || current_iteration < 0)
		{
			current_iteration++;
			iteration_start_time = elapsed_time;
			current_sub_iteration_time = 0.0;
			sub_iteration_index = 0;
			redraw = true;
		}

		if (redraw)
		{
			std::uniform_int_distribution<int> clip_range(0, int(test_clips.size() - 1));

			// Ensure that the clip changes every time so it's easier to recalibrate.
			auto next_clip_range = current_clip_index;
			while (next_clip_range == current_clip_index)
				next_clip_range = clip_range(random_engine);
			current_clip_index = next_clip_range;

			std::uniform_int_distribution<int> file_range(0, int(test_clips[current_clip_index].clips.size() - 1));
			current_test_index = file_range(random_engine);
			current_voting = 0;
		}

		auto &clip = test_clips[current_clip_index];

		if (redraw || current_sub_iteration != sub_iteration_index)
		{
			switch (sub_iteration_index)
			{
			case FirstReferenceSequence:
			case SecondReferenceSequence:
				if (!clip.clips.front().file->rewind())
					request_shutdown();
				break;

			case FirstTestSequence:
			case SecondTestSequence:
				if (clip.clips[current_test_index].codec == Codec::None)
				{
					if (!clip.clips[current_test_index].file->rewind())
						request_shutdown();
				}
				else
				{
					if (!clip.clips.front().file->rewind())
						request_shutdown();
				}
				break;

			default:
				break;
			}
		}

		current_sub_iteration = sub_iteration_index;

		YUV4MPEGFile *file = nullptr;
		switch (current_sub_iteration)
		{
		case FirstReferenceSequence:
		case SecondReferenceSequence:
			file = clip.clips.front().file.get();
			break;

		case FirstTestSequence:
		case SecondTestSequence:
			if (clip.clips[current_test_index].codec == Codec::None)
				file = clip.clips[current_test_index].file.get();
			else
				file = clip.clips.front().file.get();
			break;

		default:
			break;
		}

		if (file && !file->begin_frame())
			file = nullptr;

		for (auto &img : images.images)
		{
			cmd->image_barrier(*img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
			                   VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

			auto *ptr = cmd->update_image(*img, {}, {img->get_width(), img->get_height(), 1}, 0, 0,
			                              {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1});

			size_t size = img->get_width() * img->get_height() * (img->get_format() == VK_FORMAT_R8_UNORM ? 1 : 2);

			if (file && !file->read(ptr, size))
				file = nullptr;
			if (!file)
				memset(ptr, 0x80, size);

			cmd->image_barrier(*img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
			                   VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
			                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			                   VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
		}

		if (clip.clips[current_test_index].codec == Codec::PyroWave &&
		    (current_sub_iteration == FirstTestSequence || current_sub_iteration == SecondTestSequence))
		{
			roundtrip_pyrowave(cmd, clip.clips[current_test_index].codec_mbits);
		}

		if (file)
		{
			get_wsi().set_target_presentation_time(
				0, 1000000000ull * file->get_frame_rate_den() / file->get_frame_rate_num(), false);
		}
		else
		{
			get_wsi().set_target_presentation_time(0, 0, false);
		}
	}

	void render_frame(double, double elapsed_time) override
	{
		auto &device = get_wsi().get_device();
		auto cmd = device.request_command_buffer();

		iterate(cmd, elapsed_time);

		cmd->begin_render_pass(device.get_swapchain_render_pass(SwapchainRenderPass::Depth));
		cmd->set_sampler(0, 3, StockSampler::LinearClamp);

		auto &representative_file = *test_clips.front().clips.front().file;

		cmd->set_specialization_constant_mask(7);
		cmd->set_specialization_constant(0, representative_file.get_format() == YUV4MPEGFile::Format::YUV420P16 || representative_file.get_format() == YUV4MPEGFile::Format::YUV444P16);
		cmd->set_specialization_constant(1, representative_file.is_full_range());
		// For now, infer from limited vs full range.
		cmd->set_specialization_constant(2, !representative_file.is_full_range());

		CommandBufferUtil::setup_fullscreen_quad(*cmd, "builtin://shaders/quad.vert", "assets://yuv2rgb.frag",
		                                         {{ "DELTA", 0 }});

		const float full_color = get_wsi().get_backbuffer_format() == BackbufferFormat::HDR10 ? 0.75f : 1.0f;

		cmd->set_texture(0, 0, images.images[0]->get_view());
		cmd->set_texture(0, 1, images.images[1]->get_view());
		cmd->set_texture(0, 2, images.images[2]->get_view());
		cmd->draw(3);

		if (debug_enable)
		{
			flat_renderer.begin();
			char text[256];

			static const char *sub_iter_str[] = {
				"FirstReference (10 s)",
				"Mid-Gray 1 (3 s)",
				"FirstTest (10 s)",
				"Mid-Gray 2 (3 s)",
				"SecondReference (10 s)",
				"Mid-Gray 3 (3 s)",
				"SecondTest (10 s)",
				"Voting Mid-Gray (8 s)",
			};

			snprintf(text, sizeof(text),
					 "Iteration %d | Clip %s (%s) | %s (t = %.3f s) | voting %d",
					 current_iteration,
					 test_clips[current_clip_index].name.c_str(),
					 test_clips[current_clip_index].clips[current_test_index].desc.c_str(),
					 sub_iter_str[current_sub_iteration],
					 current_sub_iteration_time, current_voting);

			flat_renderer.render_text(GRANITE_UI_MANAGER()->get_font(UI::FontSize::Large),
									  text, vec3(20, 20, 0), vec2(400, 200),
									  vec4(full_color, full_color, 0.0f, 1.0f),
									  Font::Alignment::TopLeft);

			flat_renderer.render_text(GRANITE_UI_MANAGER()->get_font(UI::FontSize::Large),
									  text, vec3(18, 22, 0.5f), vec2(400, 200), vec4(0.0f, 0.0f, 0.0f, 1.0f),
									  Font::Alignment::TopLeft);

			flat_renderer.flush(*cmd, vec3(0), vec3(cmd->get_viewport().width, cmd->get_viewport().height, 1));
		}
		else if (current_sub_iteration == MidGrayVote)
		{
			char text[256];
			if (current_voting != 0)
				snprintf(text, sizeof(text), "Voted %d", current_voting);
			else
				strcpy(text, "Voting ...");

			flat_renderer.begin();

			flat_renderer.render_text(GRANITE_UI_MANAGER()->get_font(UI::FontSize::Huge),
			                          text, {}, vec2(cmd->get_viewport().width, cmd->get_viewport().height),
			                          vec4(full_color, full_color, 0.0f, 1.0f),
			                          Font::Alignment::Center);

			flat_renderer.render_text(GRANITE_UI_MANAGER()->get_font(UI::FontSize::Huge),
			                          text, vec3(-2.0f, 2.0f, 0.5f),
			                          vec2(cmd->get_viewport().width, cmd->get_viewport().height),
			                          vec4(0.0f, 0.0f, 0.0f, 1.0f),
			                          Font::Alignment::Center);

			flat_renderer.flush(*cmd, vec3(0), vec3(cmd->get_viewport().width, cmd->get_viewport().height, 1));
		}

		cmd->end_render_pass();

		device.submit(cmd);
	}

	YCbCrImages images;
	std::vector<TestClipGroup> test_clips;
	FlatRenderer flat_renderer;

	struct FILEDeleter { void operator()(FILE *file) { if (file) fclose(file); }};
	std::unique_ptr<FILE, FILEDeleter> evaluation_file;
};

namespace Granite
{
Application *application_create(int argc, char **argv)
{
	GRANITE_APPLICATION_SETUP_FILESYSTEM();

	if (argc != 3)
	{
		LOGE("Usage: pyrowave-evaluator test.json eval.csv\n");
		return nullptr;
	}

	try
	{
		auto *app = new EvaluatorApplication(argv[1], argv[2]);
		return app;
	}
	catch (const std::exception &e)
	{
		LOGE("application_create() threw exception: %s\n", e.what());
		return nullptr;
	}
}
}
