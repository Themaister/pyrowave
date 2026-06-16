// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

#include "context.hpp"
#include "device.hpp"
#include "image.hpp"
#include "buffer.hpp"
#include "pyrowave.h"
#include "pyrowave_decoder.hpp"
#include "pyrowave_encoder.hpp"
#include "logging.hpp"

using namespace Granite;
using namespace Vulkan;
using namespace PyroWave;

struct NullLogger : Util::LoggingInterface
{
	bool log(const char *, const char *, va_list) override
	{
		return true;
	}
};

static NullLogger null_logger;

extern "C" {
void pyrowave_get_api_version(uint32_t *major, uint32_t *minor, uint32_t *patch)
{
	*major = PYROWAVE_API_VERSION_MAJOR;
	*minor = PYROWAVE_API_VERSION_MINOR;
	*patch = PYROWAVE_API_VERSION_PATCH;
}

struct pyrowave_device_opaque
{
	Context context;
	Device device;
};

pyrowave_result pyrowave_create_device_by_compat(
	// If non-zero, needs to match VkPhysicalDeviceProperties::vendorID/deviceID.
	// Risks picking the wrong device if there are multiple ICDs for the same GPU.
	uint32_t vid, uint32_t pid,
	const pyrowave_uuid *device_uuid, // If non-NULL, needs to match VkPhysicalDeviceIDProperties::deviceUUID
	const pyrowave_uuid *driver_uuid, // If non-NULL, needs to match VkPhysicalDeviceIDProperties::driverUUID
	const pyrowave_luid *device_luid, // If non-NULL, needs to match VkPhysicalDeviceIDProperties::deviceLUID
	pyrowave_device *device)
{
	// TODO: Find a better way to do this.
	Util::set_thread_logging_interface(&null_logger);

	if (!Context::init_loader(nullptr))
		return PYROWAVE_ERROR_NO_VULKAN;

	auto *dev = new pyrowave_device_opaque();
	dev->context.set_num_thread_indices(1);
	dev->context.set_system_handles({});

	VkApplicationInfo app_info = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
	app_info.apiVersion = VK_API_VERSION_1_2;
	app_info.pApplicationName = "pyrowave-c";
	app_info.pEngineName = "Granite";
	dev->context.set_application_info(&app_info);

	// Just enable video extensions so that we can use video image usage, but don't bother creating queues for it, etc.
	if (!dev->context.init_instance(nullptr, 0, CONTEXT_CREATION_ENABLE_VIDEO_FEATURE_ONLY_BIT))
	{
		delete dev;
		return PYROWAVE_ERROR_NO_VULKAN;
	}

	uint32_t count;
	if (vkEnumeratePhysicalDevices(dev->context.get_instance(), &count, nullptr) != VK_SUCCESS)
	{
		delete dev;
		return PYROWAVE_ERROR_NO_VULKAN;
	}

	std::vector<VkPhysicalDevice> gpus(count);

	if (vkEnumeratePhysicalDevices(dev->context.get_instance(), &count, gpus.data()) < 0)
	{
		delete dev;
		return PYROWAVE_ERROR_NO_VULKAN;
	}

	VkPhysicalDevice selected_gpu = VK_NULL_HANDLE;

	for (auto &gpu : gpus)
	{
		VkPhysicalDeviceProperties props = {};
		vkGetPhysicalDeviceProperties(gpu, &props);
		// Is this even possible these days?
		if (props.apiVersion < VK_API_VERSION_1_2)
			continue;

		VkPhysicalDeviceIDProperties ids = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES };
		VkPhysicalDeviceProperties2 props2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &ids };
		vkGetPhysicalDeviceProperties2(gpu, &props2);

		if (vid && props2.properties.vendorID != vid)
			continue;
		if (pid && props2.properties.deviceID != pid)
			continue;

		if (device_uuid && memcmp(device_uuid, ids.deviceUUID, VK_UUID_SIZE) != 0)
			continue;
		if (driver_uuid && memcmp(driver_uuid, ids.driverUUID, VK_UUID_SIZE) != 0)
			continue;
		if (device_luid && !ids.deviceLUIDValid)
			continue;
		if (device_luid && memcmp(device_luid, ids.deviceLUID, VK_LUID_SIZE) != 0)
			continue;

		if (dev->context.init_device(gpu, nullptr, 0, CONTEXT_CREATION_ENABLE_VIDEO_FEATURE_ONLY_BIT))
		{
			selected_gpu = gpu;
			break;
		}
	}

	if (!selected_gpu)
	{
		delete dev;
		return PYROWAVE_ERROR_NO_VULKAN;
	}

	dev->device.set_context(dev->context);
	*device = dev;
	return PYROWAVE_SUCCESS;
}

pyrowave_result pyrowave_create_default_device(pyrowave_device *device)
{
	return pyrowave_create_device_by_compat(0, 0, nullptr, nullptr, nullptr, device);
}

void pyrowave_device_report_performance_stats(pyrowave_device device, pyrowave_message_cb cb, void *userdata, bool reset)
{
	Util::set_thread_logging_interface(&null_logger);

	device->device.timestamp_log([=](const std::string &tag, const TimestampIntervalReport &report)
	{
		char msg[256];
		snprintf(msg, sizeof(msg), "%s: %.3f ms per frame\n", tag.c_str(), report.time_per_frame_context * 1e3);
		cb(userdata, msg);
	});

	if (reset)
		device->device.timestamp_log_reset();
}

void pyrowave_device_get_vk_device_handles(
	pyrowave_device device,
	VkInstance *vk_instance, VkPhysicalDevice *vk_physical_device,
	VkDevice *vk_device)
{
	Util::set_thread_logging_interface(&null_logger);

	if (vk_instance)
		*vk_instance = device->device.get_instance();
	if (vk_physical_device)
		*vk_physical_device = device->device.get_physical_device();
	if (vk_device)
		*vk_device = device->device.get_device();
}

void pyrowave_device_destroy(pyrowave_device device)
{
	Util::set_thread_logging_interface(&null_logger);

	delete device;
}

struct pyrowave_sync_object_opaque
{
	Device *device = nullptr;
	Semaphore semaphore;
};

pyrowave_result
pyrowave_sync_object_create(const pyrowave_sync_object_create_info *info, pyrowave_sync_object *out_sync)
{
	Util::set_thread_logging_interface(&null_logger);

	if (!info->device)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	if ((info->handle_type & (
		VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT |
		VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT |
		VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT |
		VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT |
		VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT)) == 0)
	{
		return PYROWAVE_ERROR_INVALID_ARGUMENT;
	}

	auto &device = info->device->device;

	if (!device.get_device_features().supports_external)
		return PYROWAVE_ERROR_NOT_IMPLEMENTED;

	auto sem = device.request_semaphore_external(info->semaphore_type, info->handle_type);
	if (!sem)
		return PYROWAVE_ERROR_UNSUPPORTED_EXTERNAL_HANDLE;

	ExternalHandle ext = {};
	ext.handle = (decltype(ext.handle))info->external_handle;
	ext.semaphore_handle_type = info->handle_type;
	if (ext && !sem->import_from_handle(ext))
		return PYROWAVE_ERROR_FAILED_EXTERNAL_HANDLE;

	if (!ext && !(info->import_flags & VK_SEMAPHORE_IMPORT_TEMPORARY_BIT))
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	auto *sync = new pyrowave_sync_object_opaque();
	sync->device = &device;
	sync->semaphore = std::move(sem);

	*out_sync = sync;
	return PYROWAVE_SUCCESS;
}

VkSemaphore
pyrowave_sync_object_get_semaphore(pyrowave_sync_object sync)
{
	if (!sync)
		return VK_NULL_HANDLE;

	Util::set_thread_logging_interface(&null_logger);
	return sync->semaphore->get_semaphore();
}

pyrowave_result
pyrowave_sync_object_export_handle(pyrowave_sync_object sync, pyrowave_os_handle *handle)
{
	if (!sync)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	Util::set_thread_logging_interface(&null_logger);
	if (auto native_handle = sync->semaphore->export_to_handle())
	{
		*handle = (pyrowave_os_handle)native_handle.handle;
		return PYROWAVE_SUCCESS;
	}
	else
	{
		return PYROWAVE_ERROR_FAILED_EXTERNAL_HANDLE;
	}
}

void pyrowave_sync_object_destroy(pyrowave_sync_object sync)
{
	Util::set_thread_logging_interface(&null_logger);
	delete sync;
}

struct pyrowave_image_opaque
{
	Device *device = nullptr;
	ImageHandle img;
};

pyrowave_result pyrowave_image_create(const pyrowave_image_create_info *info, pyrowave_image *out_image)
{
	Util::set_thread_logging_interface(&null_logger);
	if (!info->device || !info->image_create_info)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	auto &device = info->device->device;

	if (info->image_create_info->sharingMode != VK_SHARING_MODE_EXCLUSIVE)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	if (info->handle_type == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT &&
		info->image_create_info->tiling != VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	if (info->handle_type != VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT &&
		info->image_create_info->tiling != VK_IMAGE_TILING_OPTIMAL)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	if (info->image_create_info->imageType != VK_IMAGE_TYPE_2D)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	ImageCreateInfo image_create_info = {};
	image_create_info.domain = ImageDomain::Physical;
	image_create_info.misc = IMAGE_MISC_EXTERNAL_MEMORY_BIT | IMAGE_MISC_NO_DEFAULT_VIEWS_BIT;
	image_create_info.external.handle = (decltype(image_create_info.external.handle))info->external_handle;
	image_create_info.external.memory_handle_type = info->handle_type;
	image_create_info.pnext = const_cast<void *>(info->image_create_info->pNext);
	image_create_info.layout = ImageLayout::General;

	image_create_info.type = info->image_create_info->imageType;
	image_create_info.format = info->image_create_info->format;
	image_create_info.flags = info->image_create_info->flags;
	image_create_info.width = info->image_create_info->extent.width;
	image_create_info.height = info->image_create_info->extent.height;
	image_create_info.depth = info->image_create_info->extent.depth;
	image_create_info.layers = info->image_create_info->arrayLayers;
	image_create_info.levels = info->image_create_info->mipLevels;
	image_create_info.samples = info->image_create_info->samples;
	image_create_info.usage = info->image_create_info->usage;

	if (device.get_device_features().driver_id == VK_DRIVER_ID_NVIDIA_PROPRIETARY &&
	    (info->handle_type == VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT ||
	     info->handle_type == VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT))
	{
		VkFormatProperties3 format_properties = { VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3 };
		device.get_format_properties(image_create_info.format, &format_properties);

		if (format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_VIDEO_ENCODE_INPUT_BIT_KHR)
		{
			// NVIDIA workaround. For planar formats, the D3D side assumes video compatible layouts.
			image_create_info.usage |= VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR;

			// If we're on an older driver, just pass it through as-is.
			// Normally we have to pass down a codec profile, but this is mostly noise.
			if (device.get_device_features().video_maintenance1_features.videoMaintenance1)
				image_create_info.flags |= VK_IMAGE_CREATE_VIDEO_PROFILE_INDEPENDENT_BIT_KHR;
		}
	}

	auto img = device.create_image(image_create_info);
	if (!img)
		return PYROWAVE_ERROR_FAILED_EXTERNAL_HANDLE;

	auto *image = new pyrowave_image_opaque();
	image->device = &device;
	image->img = std::move(img);

	*out_image = image;
	return PYROWAVE_SUCCESS;
}

VkImage pyrowave_image_get_handle(pyrowave_image image)
{
	Util::set_thread_logging_interface(&null_logger);
	return image->img->get_image();
}

pyrowave_result
pyrowave_image_get_image_view(pyrowave_image image, VkImageAspectFlagBits aspect,
							  VkImageUsageFlagBits usage, pyrowave_image_view *view)
{
	Util::set_thread_logging_interface(&null_logger);

	if ((aspect & (VK_IMAGE_ASPECT_PLANE_0_BIT | VK_IMAGE_ASPECT_PLANE_1_BIT | VK_IMAGE_ASPECT_PLANE_2_BIT)) == 0)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;
	if (usage != VK_IMAGE_USAGE_SAMPLED_BIT && usage != VK_IMAGE_USAGE_STORAGE_BIT)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	auto &img = *image->img;

	*view = {};
	view->image = img.get_image();
	view->image_format = img.get_format();
	view->width = img.get_width();
	view->height = img.get_height();
	view->layout = VK_IMAGE_LAYOUT_GENERAL;

	// Handle the usual suspects.
	switch (img.get_format())
	{
	// Normal explicit planar formats.
	case VK_FORMAT_R8_UNORM:
	case VK_FORMAT_R16_UNORM:
		view->aspect = VK_IMAGE_ASPECT_COLOR_BIT;
		view->swizzle = VK_COMPONENT_SWIZZLE_IDENTITY;
		view->view_format = img.get_format();
		break;

	case VK_FORMAT_R8G8_UNORM:
	case VK_FORMAT_R16G16_UNORM:
		if (usage == VK_IMAGE_USAGE_STORAGE_BIT)
			return PYROWAVE_ERROR_INVALID_ARGUMENT;
		if (aspect == VK_IMAGE_ASPECT_PLANE_0_BIT)
			return PYROWAVE_ERROR_INVALID_ARGUMENT;
		view->aspect = VK_IMAGE_ASPECT_COLOR_BIT;
		view->swizzle = aspect == VK_IMAGE_ASPECT_PLANE_2_BIT ? VK_COMPONENT_SWIZZLE_G : VK_COMPONENT_SWIZZLE_R;
		view->view_format = img.get_format();
		break;

	// Special 4:4:4 HDR10 format
	case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
		if (usage == VK_IMAGE_USAGE_STORAGE_BIT)
			return PYROWAVE_ERROR_INVALID_ARGUMENT;
		view->aspect = VK_IMAGE_ASPECT_COLOR_BIT;
		view->view_format = img.get_format();

		switch (aspect)
		{
		case VK_IMAGE_ASPECT_PLANE_0_BIT: view->swizzle = VK_COMPONENT_SWIZZLE_R; break;
		case VK_IMAGE_ASPECT_PLANE_1_BIT: view->swizzle = VK_COMPONENT_SWIZZLE_G; break;
		case VK_IMAGE_ASPECT_PLANE_2_BIT: view->swizzle = VK_COMPONENT_SWIZZLE_B; break;
		default: return PYROWAVE_ERROR_INVALID_ARGUMENT;
		}
		break;

	// 3-plane YCbCr
	case VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM:
	case VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM:
		view->view_format = VK_FORMAT_R8_UNORM;
		view->aspect = aspect;
		if (aspect != VK_IMAGE_ASPECT_PLANE_0_BIT && img.get_format() == VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM)
		{
			view->width /= 2;
			view->height /= 2;
		}
		break;

	case VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16:
	case VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16:
		view->view_format = VK_FORMAT_R10X6_UNORM_PACK16;
		view->aspect = aspect;
		if (aspect != VK_IMAGE_ASPECT_PLANE_0_BIT && img.get_format() == VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16)
		{
			view->width /= 2;
			view->height /= 2;
		}
		break;

	case VK_FORMAT_G16_B16_R16_3PLANE_420_UNORM:
	case VK_FORMAT_G16_B16_R16_3PLANE_444_UNORM:
		view->view_format = VK_FORMAT_R16_UNORM;
		view->aspect = aspect;
		if (aspect != VK_IMAGE_ASPECT_PLANE_0_BIT && img.get_format() == VK_FORMAT_G16_B16_R16_3PLANE_420_UNORM)
		{
			view->width /= 2;
			view->height /= 2;
		}
		break;

	// 2-plane YCbCr
	case VK_FORMAT_G8_B8R8_2PLANE_420_UNORM:
	case VK_FORMAT_G8_B8R8_2PLANE_444_UNORM:
		switch (aspect)
		{
		case VK_IMAGE_ASPECT_PLANE_0_BIT:
			view->view_format = VK_FORMAT_R8_UNORM;
			view->aspect = VK_IMAGE_ASPECT_PLANE_0_BIT;
			break;

		case VK_IMAGE_ASPECT_PLANE_1_BIT:
			if (usage == VK_IMAGE_USAGE_STORAGE_BIT)
				return PYROWAVE_ERROR_INVALID_ARGUMENT;
			view->swizzle = VK_COMPONENT_SWIZZLE_R;
			view->view_format = VK_FORMAT_R8G8_UNORM;
			view->aspect = VK_IMAGE_ASPECT_PLANE_1_BIT;
			if (img.get_format() == VK_FORMAT_G8_B8R8_2PLANE_420_UNORM)
			{
				view->width /= 2;
				view->height /= 2;
			}
			break;

		case VK_IMAGE_ASPECT_PLANE_2_BIT:
			if (usage == VK_IMAGE_USAGE_STORAGE_BIT)
				return PYROWAVE_ERROR_INVALID_ARGUMENT;
			view->swizzle = VK_COMPONENT_SWIZZLE_G;
			view->view_format = VK_FORMAT_R8G8_UNORM;
			view->aspect = VK_IMAGE_ASPECT_PLANE_1_BIT;
			if (img.get_format() == VK_FORMAT_G8_B8R8_2PLANE_420_UNORM)
			{
				view->width /= 2;
				view->height /= 2;
			}
			break;

		default:
			return PYROWAVE_ERROR_INVALID_ARGUMENT;
		}
		break;

	case VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16:
	case VK_FORMAT_G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16:
		switch (aspect)
		{
		case VK_IMAGE_ASPECT_PLANE_0_BIT:
			view->view_format = VK_FORMAT_R10X6_UNORM_PACK16;
			view->aspect = VK_IMAGE_ASPECT_PLANE_0_BIT;
			break;

		case VK_IMAGE_ASPECT_PLANE_1_BIT:
			if (usage == VK_IMAGE_USAGE_STORAGE_BIT)
				return PYROWAVE_ERROR_INVALID_ARGUMENT;
			view->swizzle = VK_COMPONENT_SWIZZLE_R;
			view->view_format = VK_FORMAT_R10X6G10X6_UNORM_2PACK16;
			view->aspect = VK_IMAGE_ASPECT_PLANE_1_BIT;
			if (img.get_format() == VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16)
			{
				view->width /= 2;
				view->height /= 2;
			}
			break;

		case VK_IMAGE_ASPECT_PLANE_2_BIT:
			if (usage == VK_IMAGE_USAGE_STORAGE_BIT)
				return PYROWAVE_ERROR_INVALID_ARGUMENT;
			view->swizzle = VK_COMPONENT_SWIZZLE_G;
			view->view_format = VK_FORMAT_R10X6G10X6_UNORM_2PACK16;
			view->aspect = VK_IMAGE_ASPECT_PLANE_1_BIT;
			if (img.get_format() == VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16)
			{
				view->width /= 2;
				view->height /= 2;
			}
			break;

		default:
			return PYROWAVE_ERROR_INVALID_ARGUMENT;
		}
		break;

	case VK_FORMAT_G16_B16R16_2PLANE_420_UNORM:
	case VK_FORMAT_G16_B16R16_2PLANE_444_UNORM:
		switch (aspect)
		{
		case VK_IMAGE_ASPECT_PLANE_0_BIT:
			view->view_format = VK_FORMAT_R16_UNORM;
			view->aspect = VK_IMAGE_ASPECT_PLANE_0_BIT;
			break;

		case VK_IMAGE_ASPECT_PLANE_1_BIT:
			if (usage == VK_IMAGE_USAGE_STORAGE_BIT)
				return PYROWAVE_ERROR_INVALID_ARGUMENT;
			view->swizzle = VK_COMPONENT_SWIZZLE_R;
			view->view_format = VK_FORMAT_R16G16_UNORM;
			view->aspect = VK_IMAGE_ASPECT_PLANE_1_BIT;
			if (img.get_format() == VK_FORMAT_G16_B16R16_2PLANE_420_UNORM)
			{
				view->width /= 2;
				view->height /= 2;
			}
			break;

		case VK_IMAGE_ASPECT_PLANE_2_BIT:
			if (usage == VK_IMAGE_USAGE_STORAGE_BIT)
				return PYROWAVE_ERROR_INVALID_ARGUMENT;
			view->swizzle = VK_COMPONENT_SWIZZLE_G;
			view->view_format = VK_FORMAT_R16G16_UNORM;
			view->aspect = VK_IMAGE_ASPECT_PLANE_1_BIT;
			if (img.get_format() == VK_FORMAT_G16_B16R16_2PLANE_420_UNORM)
			{
				view->width /= 2;
				view->height /= 2;
			}
			break;

		default:
			return PYROWAVE_ERROR_INVALID_ARGUMENT;
		}
		break;

	default:
		return PYROWAVE_ERROR_NOT_IMPLEMENTED;
	}

	return PYROWAVE_SUCCESS;
}

void pyrowave_image_destroy(pyrowave_image image)
{
	Util::set_thread_logging_interface(&null_logger);
	delete image;
}

struct pyrowave_encoder_opaque
{
	Device *device = nullptr;
	Encoder encoder;
	Fence queued_fence;
	BufferHandle queued_meta;
	BufferHandle queued_bitstream;
	ChromaSubsampling chroma = {};
	int width = 0;
	int height = 0;
};

pyrowave_result
pyrowave_encoder_create(const pyrowave_encoder_create_info *info, pyrowave_encoder *encoder)
{
	Util::set_thread_logging_interface(&null_logger);

	if (!info->device)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	if (info->width <= 0 || info->height <= 0)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	if (info->chroma == PYROWAVE_CHROMA_SUBSAMPLING_420 && (info->width % 2 || info->height % 2))
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	auto *enc = new pyrowave_encoder_opaque();
	enc->device = &info->device->device;
	enc->chroma = ChromaSubsampling(info->chroma);
	enc->width = info->width;
	enc->height = info->height;

	if (!enc->encoder.init(&info->device->device, info->width, info->height, enc->chroma))
	{
		delete enc;
		return PYROWAVE_ERROR_GENERIC;
	}

	*encoder = enc;
	return PYROWAVE_SUCCESS;
}

struct WrappedViewBuffers : ViewBuffers
{
	ImageHandle wrapped_images[3];
	ImageViewHandle image_views[3];
	bool wrap(Device *device, const pyrowave_gpu_buffers *buffers, VkImageUsageFlags usage);
};

bool WrappedViewBuffers::wrap(Device *device, const pyrowave_gpu_buffers *buffers, VkImageUsageFlags usage)
{
	for (int i = 0; i < 3; i++)
	{
		ImageCreateInfo image_info = {};
		image_info.usage = usage;
		image_info.type = VK_IMAGE_TYPE_2D;
		image_info.domain = ImageDomain::Physical;
		image_info.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
		image_info.width = buffers->planes[i].width;
		image_info.height = buffers->planes[i].height;
		image_info.format = buffers->planes[i].image_format;

		// The exact numbers aren't important.
		image_info.layers = buffers->planes[i].layer + 1;
		image_info.levels = buffers->planes[i].mip_level + 1;

		image_info.layout =
			buffers->planes[i].layout == VK_IMAGE_LAYOUT_GENERAL ? ImageLayout::General : ImageLayout::Optimal;
		wrapped_images[i] = device->wrap_image(image_info, buffers->planes[i].image);
		if (!wrapped_images[i])
			return false;

		ImageViewCreateInfo view_info = {};
		view_info.image = wrapped_images[i].get();
		view_info.format = buffers->planes[i].view_format;
		view_info.view_type = VK_IMAGE_VIEW_TYPE_2D;
		view_info.layers = 1;
		view_info.levels = 1;
		view_info.base_level = buffers->planes[i].mip_level;
		view_info.base_layer = buffers->planes[i].layer;
		view_info.swizzle.r = buffers->planes[i].swizzle;
		view_info.swizzle.g = VK_COMPONENT_SWIZZLE_IDENTITY;
		view_info.swizzle.b = VK_COMPONENT_SWIZZLE_IDENTITY;
		view_info.swizzle.a = VK_COMPONENT_SWIZZLE_IDENTITY;
		view_info.aspect = buffers->planes[i].aspect;
		image_views[i] = device->create_image_view(view_info);
		if (!image_views[i])
			return false;

		planes[i] = image_views[i].get();
	}

	return true;
}

static void pyrowave_device_wait_semaphore(Device *device, const pyrowave_gpu_sync_operation *acquire, VkPipelineStageFlags2 stages)
{
	if (acquire && acquire->sync.semaphore != VK_NULL_HANDLE)
	{
		auto sem = device->request_semaphore(
			acquire->sync.value != 0 ? VK_SEMAPHORE_TYPE_TIMELINE : VK_SEMAPHORE_TYPE_BINARY, acquire->sync.semaphore);
		sem->signal_external();

		if (acquire->sync.value)
		{
			sem = device->request_timeline_semaphore_as_binary(*sem, acquire->sync.value);
			sem->signal_external();
		}

		device->add_wait_semaphore(CommandBuffer::Type::Generic, std::move(sem), stages, false);
	}
}

static void pyrowave_device_signal_semaphore(Device *device, const pyrowave_gpu_sync_operation *release)
{
	if (release && release->sync.semaphore != VK_NULL_HANDLE)
	{
		auto signal = device->request_semaphore(
			release->sync.value != 0 ? VK_SEMAPHORE_TYPE_TIMELINE : VK_SEMAPHORE_TYPE_BINARY, release->sync.semaphore);

		if (release->sync.value)
			signal = device->request_timeline_semaphore_as_binary(*signal, release->sync.value);

		if (signal)
			device->submit_empty(CommandBuffer::Type::Generic, nullptr, signal.get());
	}
}

pyrowave_result
pyrowave_encoder_encode_gpu_synchronous(pyrowave_encoder encoder,
                                        const pyrowave_gpu_sync_operation *acquire,
                                        const pyrowave_gpu_sync_operation *release,
                                        const pyrowave_gpu_buffers *buffers,
                                        const pyrowave_rate_control *rate_control)
{
	Util::set_thread_logging_interface(&null_logger);
	auto *device = encoder->device;

	device->next_frame_context();

	BufferCreateInfo bufinfo = {};
	bufinfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
	                VK_BUFFER_USAGE_TRANSFER_DST_BIT |
	                VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

	bufinfo.size = encoder->encoder.get_meta_required_size();
	bufinfo.domain = BufferDomain::CachedHost;
	encoder->queued_meta = device->create_buffer(bufinfo);

	if (!encoder->queued_meta)
		return PYROWAVE_ERROR_OUT_OF_HOST_MEMORY;

	bufinfo.domain = BufferDomain::Device;
	auto queued_meta_gpu = device->create_buffer(bufinfo);

	if (!queued_meta_gpu)
		return PYROWAVE_ERROR_OUT_OF_DEVICE_MEMORY;

	auto target_bitstream_size = rate_control->maximum_bitstream_size & ~VkDeviceSize(3u);

	// Check for bogus sizes.
	if (target_bitstream_size > UINT32_MAX || target_bitstream_size == 0)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	bufinfo.size = target_bitstream_size + encoder->encoder.get_meta_required_size();
	bufinfo.domain = BufferDomain::CachedHost;
	encoder->queued_bitstream = device->create_buffer(bufinfo);

	if (!encoder->queued_bitstream)
		return PYROWAVE_ERROR_OUT_OF_HOST_MEMORY;

	bufinfo.domain = BufferDomain::Device;
	auto queued_bitstream_gpu = device->create_buffer(bufinfo);

	if (!queued_bitstream_gpu)
		return PYROWAVE_ERROR_OUT_OF_DEVICE_MEMORY;

	Encoder::BitstreamBuffers bitstream_buffers = {};

	WrappedViewBuffers views = {};
	if (!views.wrap(device, buffers, VK_IMAGE_USAGE_SAMPLED_BIT))
		return PYROWAVE_ERROR_OUT_OF_HOST_MEMORY;

	bitstream_buffers.meta.buffer = queued_meta_gpu.get();
	bitstream_buffers.meta.size = queued_meta_gpu->get_create_info().size;
	bitstream_buffers.bitstream.buffer = queued_bitstream_gpu.get();
	bitstream_buffers.bitstream.size = queued_bitstream_gpu->get_create_info().size;
	bitstream_buffers.target_size = target_bitstream_size;

	auto cmd = device->request_command_buffer();

	if (acquire)
	{
		for (size_t i = 0; i < acquire->num_images; i++)
		{
			cmd->acquire_image_barrier(*acquire->images[i].image->img,
			                           VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
			                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			                           VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
			                           acquire->images[i].queue_family_index);
		}
	}

	auto ret = encoder->encoder.encode(*cmd, views, bitstream_buffers);
	if (!ret)
	{
		device->submit_discard(cmd);
		return PYROWAVE_ERROR_INVALID_ARGUMENT;
	}

	if (release)
	{
		for (size_t i = 0; i < release->num_images; i++)
		{
			cmd->release_image_barrier(*release->images[i].image->img,
									   VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
									   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
									   VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
									   release->images[i].queue_family_index);
		}
	}

	// NVIDIA really doesn't like it if we write bitstream to cached host.
	// Performance issue since these memory types are mapped coherent on the GPU.
	// A staging copy is just better. Could avoid it on iGPU, but iGPU isn't really supposed to be
	// used as the encoder when streaming.
	cmd->copy_buffer(*encoder->queued_meta, *queued_meta_gpu);
	cmd->copy_buffer(*encoder->queued_bitstream, *queued_bitstream_gpu);

	cmd->barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
				 VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);

	pyrowave_device_wait_semaphore(device, acquire, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	encoder->queued_fence.reset();
	device->submit(cmd, &encoder->queued_fence);
	pyrowave_device_signal_semaphore(device, acquire);

	return PYROWAVE_SUCCESS;
}

pyrowave_result
pyrowave_encoder_encode_cpu_synchronous(pyrowave_encoder encoder, const pyrowave_cpu_buffer *buffers,
										const pyrowave_rate_control *rate_control)
{
	Util::set_thread_logging_interface(&null_logger);
	int num_planes = buffers->format == PYROWAVE_CPU_BUFFER_FORMAT_NV12 ? 2 : 3;
	auto *device = encoder->device;
	ImageHandle images[3];

	// Validate some assumptions.
	if (buffers->width != encoder->width || buffers->height != encoder->height)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	if (encoder->chroma == ChromaSubsampling::Chroma420 && buffers->format == PYROWAVE_CPU_BUFFER_FORMAT_YUV444P)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;
	if (encoder->chroma == ChromaSubsampling::Chroma444 && buffers->format != PYROWAVE_CPU_BUFFER_FORMAT_YUV444P)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	for (int plane = 0; plane < num_planes; plane++)
	{
		int plane_width = encoder->width;
		int plane_height = encoder->height;

		if (plane != 0 && encoder->chroma == ChromaSubsampling::Chroma420)
		{
			plane_width /= 2;
			plane_height /= 2;
		}

		const size_t plane_bpp = num_planes == 2 && plane == 1 ? 2 : 1;

		if (buffers->row_stride_in_bytes[plane] < plane_width * plane_bpp)
			return PYROWAVE_ERROR_INVALID_ARGUMENT;
		if (buffers->row_stride_in_bytes[plane] * plane_height > buffers->plane_size_in_bytes[plane])
			return PYROWAVE_ERROR_INVALID_ARGUMENT;
	}

	for (int plane = 0; plane < num_planes; plane++)
	{
		unsigned plane_bpp = num_planes == 2 && plane == 1 ? 2 : 1;

		const ImageInitialData initial = {
			buffers->data[plane],
			uint32_t(buffers->row_stride_in_bytes[plane] / plane_bpp)
		};

		auto info = ImageCreateInfo::immutable_2d_image(
			buffers->width, buffers->height,
			plane_bpp == 2 ? VK_FORMAT_R8G8_UNORM : VK_FORMAT_R8_UNORM);

		if (plane != 0 && encoder->chroma == ChromaSubsampling::Chroma420)
		{
			info.width /= 2;
			info.height /= 2;
		}

		images[plane] = device->create_image(info, &initial);
		if (!images[plane])
			return PYROWAVE_ERROR_OUT_OF_DEVICE_MEMORY;
	}

	pyrowave_gpu_buffers gpu_buffers = {};

	for (int plane = 0; plane < 3; plane++)
	{
		auto &p = gpu_buffers.planes[plane];
		p.width = images[plane] ? images[plane]->get_width() : images[1]->get_width();
		p.height = images[plane] ? images[plane]->get_height() : images[1]->get_height();

		p.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
		p.swizzle = num_planes == 2 && plane == 2 ? VK_COMPONENT_SWIZZLE_G : VK_COMPONENT_SWIZZLE_R;
		p.image_format = images[plane] ? images[plane]->get_format() : VK_FORMAT_R8G8_UNORM;
		p.view_format = p.image_format;
		p.layout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
		p.image = images[plane] ? images[plane]->get_image() : images[1]->get_image();
	}

	auto ret = pyrowave_encoder_encode_gpu_synchronous(encoder, nullptr, nullptr, &gpu_buffers, rate_control);
	return ret;
}

pyrowave_result
pyrowave_encoder_compute_num_packets(pyrowave_encoder encoder, size_t packet_boundary, size_t *num_packets)
{
	Util::set_thread_logging_interface(&null_logger);
	if (!encoder->queued_fence)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;
	encoder->queued_fence->wait();

	auto *mapped_meta = encoder->device->map_host_buffer(*encoder->queued_meta, MEMORY_ACCESS_READ_BIT);
	*num_packets = encoder->encoder.compute_num_packets(mapped_meta, packet_boundary);
	return PYROWAVE_SUCCESS;
}

pyrowave_result
pyrowave_encoder_packetize(pyrowave_encoder encoder, pyrowave_packet *packets, size_t packet_boundary,
                           size_t *out_packets, void *bitstream, size_t size)
{
	Util::set_thread_logging_interface(&null_logger);
	if (!encoder->queued_fence)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;
	encoder->queued_fence->wait();

	auto *mapped_meta = encoder->device->map_host_buffer(*encoder->queued_meta, MEMORY_ACCESS_READ_BIT);
	auto *mapped_bitstream = encoder->device->map_host_buffer(*encoder->queued_bitstream, MEMORY_ACCESS_READ_BIT);

	*out_packets = encoder->encoder.packetize(
		reinterpret_cast<Encoder::Packet *>(packets), packet_boundary, bitstream,
		size, mapped_meta, mapped_bitstream);

	return PYROWAVE_SUCCESS;
}

void pyrowave_encoder_destroy(pyrowave_encoder encoder)
{
	Util::set_thread_logging_interface(&null_logger);
	delete encoder;
}

struct pyrowave_decoder_opaque
{
	Device *device = nullptr;
	Decoder decoder;
	ImageHandle planes[3];
	bool fragment_path = false;
	ChromaSubsampling chroma = {};
	int width = 0;
	int height = 0;
};

bool pyrowave_decoder_device_prefers_fragment_path(pyrowave_device device)
{
	Util::set_thread_logging_interface(&null_logger);
	return Decoder::device_prefers_fragment_path(device->device);
}

pyrowave_result
pyrowave_decoder_create(const pyrowave_decoder_create_info *info, pyrowave_decoder *decoder)
{
	Util::set_thread_logging_interface(&null_logger);
	if (!info->device)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	if (info->width <= 0 || info->height <= 0)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	if (info->chroma == PYROWAVE_CHROMA_SUBSAMPLING_420 && (info->width % 2 || info->height % 2))
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	auto *dec = new pyrowave_decoder_opaque();
	dec->device = &info->device->device;
	dec->chroma = ChromaSubsampling(info->chroma);
	dec->fragment_path = info->fragment_path;
	dec->width = info->width;
	dec->height = info->height;

	if (!dec->decoder.init(dec->device, info->width, info->height, dec->chroma, info->fragment_path))
	{
		delete dec;
		return PYROWAVE_ERROR_INVALID_ARGUMENT;
	}

	*decoder = dec;
	return PYROWAVE_SUCCESS;
}

void pyrowave_decoder_clear(pyrowave_decoder decoder)
{
	Util::set_thread_logging_interface(&null_logger);
	decoder->decoder.clear();
}

// A frame is potentially split into multiple packets.
pyrowave_result
pyrowave_decoder_push_packet(pyrowave_decoder decoder, const void *data, size_t size)
{
	Util::set_thread_logging_interface(&null_logger);
	bool ret = decoder->decoder.push_packet(data, size);
	return ret ? PYROWAVE_SUCCESS : PYROWAVE_ERROR_INVALID_ARGUMENT;
}

// For error correction purposes, it may be okay to decode a frame which dropped some packets.
bool pyrowave_decoder_decode_is_ready(pyrowave_decoder decoder, bool allow_partial_frame)
{
	Util::set_thread_logging_interface(&null_logger);
	return decoder->decoder.decode_is_ready(allow_partial_frame);
}

pyrowave_result
pyrowave_decoder_decode_gpu_buffer(pyrowave_decoder decoder,
                                   const pyrowave_gpu_sync_operation *acquire,
                                   const pyrowave_gpu_sync_operation *release,
                                   const pyrowave_gpu_buffers *buffers)
{
	Util::set_thread_logging_interface(&null_logger);
	auto *device = decoder->device;
	device->next_frame_context();

	WrappedViewBuffers views = {};
	if (!views.wrap(device, buffers, decoder->fragment_path ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT : VK_IMAGE_USAGE_STORAGE_BIT))
		return PYROWAVE_ERROR_OUT_OF_HOST_MEMORY;

	// Just use normal graphics queue here since the result will likely be consumed there.
	auto cmd = device->request_command_buffer();

	VkPipelineStageFlags2 stages = decoder->fragment_path
		                               ? VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
		                               : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

	VkAccessFlags2 access = decoder->fragment_path
		                        ? VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
		                        : VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

	if (acquire)
	{
		for (size_t i = 0; i < acquire->num_images; i++)
		{
			if (acquire->images[i].queue_family_index != VK_QUEUE_FAMILY_IGNORED)
			{
				cmd->acquire_image_barrier(*acquire->images[i].image->img,
										   VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
										   stages, access, acquire->images[i].queue_family_index);
			}
			else
			{
				cmd->image_barrier(*acquire->images[i].image->img,
								   VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
								   stages, 0, stages, access);
			}
		}
	}

	auto ret = decoder->decoder.decode(*cmd, views);
	if (!ret)
	{
		device->submit_discard(cmd);
		return PYROWAVE_ERROR_INVALID_ARGUMENT;
	}

	if (release)
	{
		for (size_t i = 0; i < release->num_images; i++)
		{
			cmd->release_image_barrier(*release->images[i].image->img,
									   VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
									   stages, access, release->images[i].queue_family_index);
		}
	}

	pyrowave_device_wait_semaphore(device, acquire, stages);
	// This just queues up a command buffer, flush only happens when sync objects are signaled.
	device->submit(cmd);
	pyrowave_device_signal_semaphore(device, release);

	return PYROWAVE_SUCCESS;
}

pyrowave_result
pyrowave_decoder_decode_cpu_buffer_synchronous(pyrowave_decoder decoder, const pyrowave_cpu_buffer *buffers)
{
	Util::set_thread_logging_interface(&null_logger);
	auto *device = decoder->device;

	if (buffers->width != decoder->width || buffers->height != decoder->height)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	if (buffers->format == PYROWAVE_CPU_BUFFER_FORMAT_NV12)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	if (decoder->chroma == ChromaSubsampling::Chroma420 && buffers->format == PYROWAVE_CPU_BUFFER_FORMAT_YUV444P)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;
	if (decoder->chroma == ChromaSubsampling::Chroma444 && buffers->format != PYROWAVE_CPU_BUFFER_FORMAT_YUV444P)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	for (int plane = 0; plane < 3; plane++)
	{
		int plane_width = decoder->width;
		int plane_height = decoder->height;

		if (plane != 0 && decoder->chroma == ChromaSubsampling::Chroma420)
		{
			plane_width /= 2;
			plane_height /= 2;
		}

		const size_t plane_bpp = 1;

		if (buffers->row_stride_in_bytes[plane] < plane_width * plane_bpp)
			return PYROWAVE_ERROR_INVALID_ARGUMENT;
		if (buffers->row_stride_in_bytes[plane] * plane_height > buffers->plane_size_in_bytes[plane])
			return PYROWAVE_ERROR_INVALID_ARGUMENT;
	}

	for (int plane = 0; plane < 3; plane++)
	{
		auto &img = decoder->planes[plane];

		if (!img)
		{
			auto info = ImageCreateInfo::immutable_2d_image(buffers->width, buffers->height, VK_FORMAT_R8_UNORM);
			info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
			if (decoder->fragment_path)
			{
				info.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
			}
			else
			{
				info.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
				info.initial_layout = VK_IMAGE_LAYOUT_GENERAL;
				info.layout = ImageLayout::General;
			}

			if (plane != 0 && decoder->chroma == ChromaSubsampling::Chroma420)
			{
				info.width /= 2;
				info.height /= 2;
			}

			img = device->create_image(info);
			if (!img)
				return PYROWAVE_ERROR_OUT_OF_DEVICE_MEMORY;
		}
	}

	if (decoder->fragment_path)
	{
		auto cmd = device->request_command_buffer();
		cmd->begin_barrier_batch();
		for (auto &img : decoder->planes)
		{
			cmd->image_barrier(*img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
				VK_PIPELINE_STAGE_2_COPY_BIT, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR,
				VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
		}
		cmd->end_barrier_batch();

		// This just queues up a command buffer, flush only happens when sync objects are signaled.
		device->submit(cmd);
	}

	pyrowave_gpu_buffers gpu_buffers = {};
	for (int plane = 0; plane < 3; plane++)
	{
		auto &p = gpu_buffers.planes[plane];
		p.image = decoder->planes[plane]->get_image();
		p.width = decoder->planes[plane]->get_width();
		p.height = decoder->planes[plane]->get_height();
		p.image_format = decoder->planes[plane]->get_format();
		p.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
		p.swizzle = VK_COMPONENT_SWIZZLE_IDENTITY;
		p.view_format = p.image_format;
		p.layout = decoder->fragment_path ? VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
	}

	BufferCreateInfo bufinfo = {};
	BufferHandle readback_buffers[3];

	auto res = pyrowave_decoder_decode_gpu_buffer(decoder, nullptr, nullptr, &gpu_buffers);
	if (res != PYROWAVE_SUCCESS)
		return res;

	auto cmd = device->request_command_buffer();

	if (decoder->fragment_path)
	{
		cmd->begin_barrier_batch();
		for (auto &img : decoder->planes)
		{
			cmd->image_barrier(*img, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
				VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
		}
		cmd->end_barrier_batch();
	}
	else
	{
		cmd->barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
		             VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
	}

	for (int plane = 0; plane < 3; plane++)
	{
		bufinfo.size = buffers->plane_size_in_bytes[plane];
		bufinfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		bufinfo.domain = BufferDomain::CachedHost;
		readback_buffers[plane] = device->create_buffer(bufinfo);

		cmd->copy_image_to_buffer(*readback_buffers[plane], *decoder->planes[plane], 0, {},
		                          {decoder->planes[plane]->get_width(), decoder->planes[plane]->get_height(), 1},
		                          buffers->row_stride_in_bytes[plane], 0,
		                          {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1});
	}

	cmd->barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
	             VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);

	Fence fence;
	device->submit(cmd, &fence);
	fence->wait();

	for (int plane = 0; plane < 3; plane++)
	{
		void *mapped = device->map_host_buffer(*readback_buffers[plane], MEMORY_ACCESS_READ_BIT);
		memcpy(buffers->data[plane], mapped, buffers->plane_size_in_bytes[plane]);
	}

	return PYROWAVE_SUCCESS;
}

void pyrowave_decoder_destroy(pyrowave_decoder decoder)
{
	Util::set_thread_logging_interface(&null_logger);
	delete decoder;
}
}
