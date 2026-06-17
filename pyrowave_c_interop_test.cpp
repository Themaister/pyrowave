// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

#include "device.hpp"
#include "context.hpp"
#include "image.hpp"

#include "pyrowave.h"
#include <stdio.h>
#include <cstdlib>
#include <exception>
#include <vector>

using namespace Vulkan;

#define ASSERT_THAT(x) do { \
	if (!(x)) { fprintf(stderr, "Fatal error executing %s at line %d.\n", #x, __LINE__); std::terminate(); } \
} while(false)

#define CHECKED(x) do { \
	pyrowave_result res = x; \
	if (res != PYROWAVE_SUCCESS) { fprintf(stderr, "Got pyrowave result %d while executing %s at line %d.\n", res, #x, __LINE__); std::terminate(); } \
} while(false)

static pyrowave_device create_device_from_granite(Device &device)
{
	// Verify that we can create a device from UUID/LUID.
	VkPhysicalDeviceIDProperties ids = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES };
	VkPhysicalDeviceProperties2 props2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &ids };
	vkGetPhysicalDeviceProperties2(device.get_physical_device(), &props2);

	pyrowave_uuid device_uuid, driver_uuid;
	pyrowave_luid device_luid;

	memcpy(device_uuid.uuid, ids.deviceUUID, VK_UUID_SIZE);
	memcpy(driver_uuid.uuid, ids.driverUUID, VK_UUID_SIZE);
	memcpy(device_luid.luid, ids.deviceLUID, VK_LUID_SIZE);

	pyrowave_device pyro_device;
	CHECKED(pyrowave_create_device_by_compat(device.get_gpu_properties().vendorID,
		device.get_gpu_properties().deviceID, &device_uuid, &driver_uuid,
		ids.deviceLUIDValid ? &device_luid : nullptr, &pyro_device));

	return pyro_device;
}

static pyrowave_sync_object create_sync_object_from_timeline(pyrowave_device device, SemaphoreHolder &sem)
{
	auto exported_timeline = sem.export_to_handle();
	ASSERT_THAT(exported_timeline);

	pyrowave_sync_object_create_info sync_info = {};
	sync_info.device = device;
	sync_info.handle_type = exported_timeline.semaphore_handle_type;
	sync_info.semaphore_type = VK_SEMAPHORE_TYPE_TIMELINE;
	sync_info.external_handle = (pyrowave_os_handle)exported_timeline.handle;
	pyrowave_sync_object imported_timeline;
	CHECKED(pyrowave_sync_object_create(&sync_info, &imported_timeline));
	return imported_timeline;
}

static pyrowave_sync_object create_sync_object_from_binary(pyrowave_device device, SemaphoreHolder &sem)
{
	auto exported_timeline = sem.export_to_handle();
	ASSERT_THAT(exported_timeline);

	pyrowave_sync_object_create_info sync_info = {};
	sync_info.device = device;
	sync_info.handle_type = exported_timeline.semaphore_handle_type;
	sync_info.semaphore_type = VK_SEMAPHORE_TYPE_BINARY;
	sync_info.external_handle = (pyrowave_os_handle)exported_timeline.handle;
	sync_info.import_flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT;
	pyrowave_sync_object imported_timeline;
	CHECKED(pyrowave_sync_object_create(&sync_info, &imported_timeline));
	return imported_timeline;
}

static pyrowave_image create_imported_image(pyrowave_device device, Image &img)
{
	auto exported = img.export_handle();
	ASSERT_THAT(exported);

	VkImageCreateInfo image_create_info = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
	image_create_info.flags = img.get_create_info().flags;
	image_create_info.usage = img.get_create_info().usage;
	image_create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
	image_create_info.format = img.get_create_info().format;
	image_create_info.samples = img.get_create_info().samples;
	image_create_info.mipLevels = img.get_create_info().levels;
	image_create_info.arrayLayers = img.get_create_info().layers;
	image_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	image_create_info.imageType = img.get_create_info().type;
	image_create_info.extent = { img.get_width(), img.get_height(), img.get_depth() };

	pyrowave_image_create_info image_info = {};
	image_info.device = device;
	image_info.handle_type = exported.memory_handle_type;
	image_info.external_handle = (pyrowave_os_handle)exported.handle;
	image_info.image_create_info = &image_create_info;

	pyrowave_image imported_image;
	CHECKED(pyrowave_image_create(&image_info, &imported_image));
	return imported_image;
}

static uint8_t mirror(int v)
{
	v &= 511;
	if (v > 255)
		v = 511 - v;
	ASSERT_THAT(v >= 0 && v <= 255);
	return uint8_t(v);
}

static ImageHandle create_exportable_test_image(Device &device, VkExternalMemoryHandleTypeFlagBits handle_type,
                                                VkFormat format)
{
	auto info = ImageCreateInfo::immutable_2d_image(1280, 720, format);
	info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
	             VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	info.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
	info.misc = IMAGE_MISC_NO_DEFAULT_VIEWS_BIT | IMAGE_MISC_EXTERNAL_MEMORY_BIT;
	info.layout = ImageLayout::General;
	info.initial_layout = VK_IMAGE_LAYOUT_GENERAL;
	info.external.memory_handle_type = handle_type;

	auto img = device.create_image(info);

	if (format == VK_FORMAT_G8_B8R8_2PLANE_420_UNORM)
	{
		auto cmd = device.request_command_buffer();

		auto *luma = static_cast<uint8_t *>(cmd->update_image(*img, {}, { 1280, 720, 1 }, 1280, 720, { VK_IMAGE_ASPECT_PLANE_0_BIT, 0, 0, 1 }));
		auto *chroma = static_cast<uint16_t *>(cmd->update_image(*img, {}, { 640, 360, 1 }, 640, 360, { VK_IMAGE_ASPECT_PLANE_1_BIT, 0, 0, 1 }));

		for (int y = 0; y < 720; y++)
			for (int x = 0; x < 1280; x++)
				luma[y * 1280 + x] = mirror(y * 3 + 5 * x);

		for (int y = 0; y < 360; y++)
			for (int x = 0; x < 640; x++)
				chroma[y * 640 + x] = (uint16_t(mirror(y * 7 + 5 * x)) << 8) | mirror(y + 3 * x);

		device.submit(cmd);
	}

	return img;
}

static constexpr size_t BitstreamSize = 1000000;

static void send_granite_image_to_encoder(Device &device, Image &granite_image, pyrowave_image pyro_image,
                                          SemaphoreHolder &granite_sync, pyrowave_sync_object pyro_sync_acquire, uint64_t acquire_value,
                                          pyrowave_sync_object pyro_sync_release, uint64_t release_value,
                                          pyrowave_encoder encoder)
{
	auto cmd = device.request_command_buffer();
	cmd->release_image_barrier(granite_image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
		VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_QUEUE_FAMILY_EXTERNAL);
	device.submit(cmd);

	if (acquire_value)
	{
		auto signal = device.request_timeline_semaphore_as_binary(granite_sync, acquire_value);
		device.submit_empty(CommandBuffer::Type::Generic, nullptr, signal.get());
	}

	pyrowave_gpu_external_reference ref = { pyro_image, VK_QUEUE_FAMILY_EXTERNAL };

	pyrowave_gpu_sync_operation acquire = {};
	pyrowave_gpu_sync_operation release = {};
	pyrowave_gpu_buffers buffers = {};
	pyrowave_rate_control rate_control = { BitstreamSize };

	CHECKED(pyrowave_image_get_image_view(pyro_image,
		VK_IMAGE_ASPECT_PLANE_0_BIT, VK_IMAGE_USAGE_SAMPLED_BIT, &buffers.planes[0]));
	CHECKED(pyrowave_image_get_image_view(pyro_image,
		VK_IMAGE_ASPECT_PLANE_1_BIT, VK_IMAGE_USAGE_SAMPLED_BIT, &buffers.planes[1]));
	CHECKED(pyrowave_image_get_image_view(pyro_image,
		VK_IMAGE_ASPECT_PLANE_2_BIT, VK_IMAGE_USAGE_SAMPLED_BIT, &buffers.planes[2]));

	acquire.num_images = 1;
	acquire.images = &ref;
	acquire.sync.semaphore = pyrowave_sync_object_get_semaphore(pyro_sync_acquire);
	acquire.sync.value = acquire_value;

	release.num_images = 1;
	release.images = &ref;
	release.sync.semaphore = pyrowave_sync_object_get_semaphore(pyro_sync_release);
	release.sync.value = release_value;

	CHECKED(pyrowave_encoder_encode_gpu_synchronous(encoder, &acquire, &release, &buffers, &rate_control));

	// Verify that the release value was signaled properly in finite time.
	VkSemaphoreWaitInfo wait_info = { VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
	wait_info.pSemaphores = &granite_sync.get_semaphore();
	wait_info.semaphoreCount = 1;
	wait_info.pValues = &release_value;
	auto vr = device.get_device_table().vkWaitSemaphores(device.get_device(), &wait_info, 1000000000);
	ASSERT_THAT(vr == VK_SUCCESS);
}

static void decode_image(pyrowave_image pyro_image,
                         pyrowave_sync_object pyro_sync, uint64_t release_value,
                         pyrowave_decoder decoder)
{
	// Discard the image.
	pyrowave_gpu_external_reference acquire_ref = { pyro_image, VK_QUEUE_FAMILY_IGNORED };
	pyrowave_gpu_external_reference release_ref = { pyro_image, VK_QUEUE_FAMILY_EXTERNAL };

	pyrowave_gpu_sync_operation acquire = {};
	pyrowave_gpu_sync_operation release = {};
	pyrowave_gpu_buffers buffers = {};

	acquire.num_images = 1;
	acquire.images = &acquire_ref;

	release.num_images = 1;
	release.images = &release_ref;
	release.sync.semaphore = pyrowave_sync_object_get_semaphore(pyro_sync);
	release.sync.value = release_value;

	CHECKED(pyrowave_image_get_image_view(pyro_image,
		VK_IMAGE_ASPECT_PLANE_0_BIT, VK_IMAGE_USAGE_STORAGE_BIT, &buffers.planes[0]));
	CHECKED(pyrowave_image_get_image_view(pyro_image,
		VK_IMAGE_ASPECT_PLANE_1_BIT, VK_IMAGE_USAGE_STORAGE_BIT, &buffers.planes[1]));
	CHECKED(pyrowave_image_get_image_view(pyro_image,
		VK_IMAGE_ASPECT_PLANE_2_BIT, VK_IMAGE_USAGE_STORAGE_BIT, &buffers.planes[2]));

	CHECKED(pyrowave_decoder_decode_gpu_buffer(decoder, &acquire, &release, &buffers));
}

static void send_payload_to_decoder(pyrowave_encoder encoder, pyrowave_decoder decoder)
{
	size_t num_packets;
	CHECKED(pyrowave_encoder_compute_num_packets(encoder, BitstreamSize, &num_packets));
	ASSERT_THAT(num_packets == 1);

	pyrowave_packet packet;
	std::unique_ptr<uint8_t[]> bitstream(new uint8_t[BitstreamSize]);
	CHECKED(pyrowave_encoder_packetize(encoder, &packet, BitstreamSize, &num_packets, bitstream.get(), BitstreamSize));
	CHECKED(pyrowave_decoder_push_packet(decoder, bitstream.get() + packet.offset, packet.size));
	ASSERT_THAT(pyrowave_decoder_decode_is_ready(decoder, false));
}

static void validate_mirror_buffer(Device &device, Buffer &buf, int width, int height, int dx, int dy)
{
	auto *ptr = static_cast<const uint8_t *>(device.map_host_buffer(buf, MEMORY_ACCESS_READ_BIT));

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int reference = mirror(x * dx + y * dy);
			int value = ptr[y * width + x];
			int d = std::abs(reference - value);
			ASSERT_THAT(d <= 1);
		}
	}
}

static void validate_granite_image(Device &device, Image &img, SemaphoreHolder &sem, uint64_t acquire_value)
{
	auto wait_sem = device.request_timeline_semaphore_as_binary(sem, acquire_value);
	wait_sem->signal_external();
	device.add_wait_semaphore(CommandBuffer::Type::Generic, std::move(wait_sem), VK_PIPELINE_STAGE_2_COPY_BIT, true);

	auto cmd = device.request_command_buffer();

	BufferCreateInfo bufinfo = {};
	bufinfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	bufinfo.size = img.get_width() * img.get_height();
	bufinfo.domain = BufferDomain::CachedHost;

	auto y = device.create_buffer(bufinfo);

	bufinfo.size /= 4;
	auto cb = device.create_buffer(bufinfo);
	auto cr = device.create_buffer(bufinfo);

	cmd->acquire_image_barrier(img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
	                           VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT, VK_QUEUE_FAMILY_EXTERNAL);

	cmd->copy_image_to_buffer(*y, img, 0, {}, { img.get_width(), img.get_height(), 1 }, 1280, 720,
		{ VK_IMAGE_ASPECT_PLANE_0_BIT, 0, 0, 1 });
	cmd->copy_image_to_buffer(*cb, img, 0, {}, { img.get_width() / 2, img.get_height() / 2, 1 }, 640, 360,
		{ VK_IMAGE_ASPECT_PLANE_1_BIT, 0, 0, 1 });
	cmd->copy_image_to_buffer(*cr, img, 0, {}, { img.get_width() / 2, img.get_height() / 2, 1 }, 640, 360,
		{ VK_IMAGE_ASPECT_PLANE_2_BIT, 0, 0, 1 });

	cmd->barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
	             VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_WRITE_BIT);

	Fence fence;
	device.submit(cmd, &fence);
	fence->wait();

	validate_mirror_buffer(device, *y, 1280, 720, 5, 3);
	validate_mirror_buffer(device, *cb, 640, 360, 3, 1);
	validate_mirror_buffer(device, *cr, 640, 360, 5, 7);
}

// Most basic interop scenario, OPAQUE_FD for everything.
static void test_opaque_interop()
{
	ASSERT_THAT(Context::init_loader(nullptr));

	Context ctx;
	ctx.set_num_thread_indices(1);
	ctx.set_system_handles({});
	ASSERT_THAT(ctx.init_instance_and_device(nullptr, 0, nullptr, 0));

	Device device;
	device.set_context(ctx);

	pyrowave_device pyro_device = create_device_from_granite(device);

	auto timeline_sem = device.request_semaphore_external(VK_SEMAPHORE_TYPE_TIMELINE,
		ExternalHandle::get_opaque_semaphore_handle_type());
	ASSERT_THAT(timeline_sem);
	pyrowave_sync_object imported_timeline = create_sync_object_from_timeline(pyro_device, *timeline_sem);

	auto exportable_nv12_image = create_exportable_test_image(
		device, ExternalHandle::get_opaque_memory_handle_type(),
		VK_FORMAT_G8_B8R8_2PLANE_420_UNORM);
	auto exportable_yuv420p_image = create_exportable_test_image(
		device, ExternalHandle::get_opaque_memory_handle_type(),
		VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM);

	pyrowave_image imported_nv12_image = create_imported_image(pyro_device, *exportable_nv12_image);
	pyrowave_image imported_yuv420p_image = create_imported_image(pyro_device, *exportable_yuv420p_image);

	auto binary_sem = device.request_semaphore_external(
		VK_SEMAPHORE_TYPE_BINARY, ExternalHandle::get_opaque_semaphore_handle_type());
	ASSERT_THAT(binary_sem);
	device.submit_empty(CommandBuffer::Type::Generic, nullptr, binary_sem.get());
	pyrowave_sync_object imported_binary = create_sync_object_from_binary(pyro_device, *binary_sem);

	pyrowave_encoder_create_info encoder_info = {};
	encoder_info.device = pyro_device;
	encoder_info.width = 1280;
	encoder_info.height = 720;
	encoder_info.chroma = PYROWAVE_CHROMA_SUBSAMPLING_420;
	pyrowave_encoder encoder;
	CHECKED(pyrowave_encoder_create(&encoder_info, &encoder));

	// Test the two ways we can send a sync payload, binary + temporary or persistent timeline.
	// Verify it doesn't explode.
	send_granite_image_to_encoder(device, *exportable_nv12_image, imported_nv12_image,
	                              *timeline_sem, imported_timeline, 1,
	                              imported_timeline, 2,
	                              encoder);

	send_granite_image_to_encoder(device, *exportable_nv12_image, imported_nv12_image,
	                              *timeline_sem, imported_binary, 0,
	                              imported_timeline, 3,
	                              encoder);

	// Pyrowave (or rather, Granite) API destroys objects in a deferred way.
	pyrowave_sync_object_destroy(imported_binary);

	pyrowave_decoder_create_info decoder_info = {};
	decoder_info.device = pyro_device;
	decoder_info.width = 1280;
	decoder_info.height = 720;
	decoder_info.chroma = PYROWAVE_CHROMA_SUBSAMPLING_420;
	pyrowave_decoder decoder;
	CHECKED(pyrowave_decoder_create(&decoder_info, &decoder));

	send_payload_to_decoder(encoder, decoder);
	decode_image(imported_yuv420p_image, imported_timeline, 4, decoder);
	validate_granite_image(device, *exportable_yuv420p_image, *timeline_sem, 4);

	pyrowave_sync_object_destroy(imported_timeline);
	pyrowave_image_destroy(imported_nv12_image);
	pyrowave_image_destroy(imported_yuv420p_image);
	pyrowave_encoder_destroy(encoder);
	pyrowave_decoder_destroy(decoder);
	pyrowave_device_destroy(pyro_device);
}

int main()
{
	test_opaque_interop();
}
