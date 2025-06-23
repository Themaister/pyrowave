#include "device.hpp"
#include "context.hpp"
#include "global_managers_init.hpp"
#include "pyrowave_encoder.hpp"
#include "pyrowave_decoder.hpp"
#include "filesystem.hpp"
#include <cmath>

#include "dxgi1_6.h"
#include "d3d12.h"

#include <SDL3/SDL.h>

using namespace Vulkan;

template <typename T>
class ComPtr
{
public:
	ComPtr() = default;
	~ComPtr() { release(); }

	ComPtr(const ComPtr &other) { *this = other; }
	ComPtr &operator=(const ComPtr &other);
	ComPtr &operator=(ComPtr &&other) noexcept;

	ComPtr(ComPtr &&other) noexcept { *this = std::move(other); }
	ComPtr &operator=(T *ptr_) { release(); ptr = ptr_; return *this; }

	T *operator->() { return ptr; }
	T *get() { return ptr; }

	T **operator&() { release(); return &ptr; }
	explicit operator bool() const { return ptr != nullptr; }

private:
	T *ptr = nullptr;
	void release()
	{
		if (ptr)
			ptr->Release();
		ptr = nullptr;
	}
	const ComPtr *self_addr() const { return this; }
};

template <typename T>
ComPtr<T> &ComPtr<T>::operator=(const ComPtr &other)
{
	if (this == other.self_addr())
		return *this;
	other.ptr->AddRef();
	release();
	ptr = other.ptr;
	return *this;
}

template <typename T>
ComPtr<T> &ComPtr<T>::operator=(ComPtr &&other) noexcept
{
	if (this == other.self_addr())
		return *this;
	release();
	ptr = other.ptr;
	other.ptr = nullptr;
	return *this;
}

struct DXGIContext
{
	ComPtr<IDXGIFactory> factory;
	ComPtr<IDXGIAdapter> adapter;
};

static DXGIContext query_adapter()
{
	ComPtr<IDXGIFactory> factory;
	auto hr = CreateDXGIFactory(IID_PPV_ARGS(&factory));

	if (FAILED(hr))
	{
		LOGE("Failed to create DXGI factory.\n");
		return {};
	}

	ComPtr<IDXGIAdapter> adapter;
	for (unsigned i = 0; !adapter; i++)
	{
		hr = factory->EnumAdapters(i, &adapter);
		if (hr == DXGI_ERROR_NOT_FOUND)
			break;

		ComPtr<IDXGIAdapter1> adapter1;
		if (FAILED(adapter->QueryInterface(&adapter1)))
		{
			adapter->Release();
			adapter = nullptr;
			continue;
		}

		DXGI_ADAPTER_DESC1 adapter_desc;
		adapter1->GetDesc1(&adapter_desc);
		if ((adapter_desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0)
		{
			adapter = {};
		}
	}

	return { factory, adapter };
}

static constexpr int Width = 1024;
static constexpr int Height = 1200;

struct D3DContext
{
	ComPtr<ID3D12Device> dev;
	ComPtr<ID3D12CommandQueue> queue;
	ComPtr<ID3D12CommandAllocator> allocator[2];
	ComPtr<ID3D12GraphicsCommandList> list[2];
	DXGIContext dxgi;
	LUID luid;

	ComPtr<ID3D12Resource> back_buffers[2];
	uint64_t wait_timeline[2];
	ComPtr<IDXGISwapChain3> swapchain;

	ComPtr<ID3D12Resource> texture;
	ComPtr<ID3D12Resource> texture_nv12;
	ComPtr<ID3D12Fence> fence;
};

static D3DContext create_d3d12_device()
{
	auto dxgi_context = query_adapter();
	if (!dxgi_context.adapter)
		return {};

	D3DContext ctx = {};

	auto hr = D3D12CreateDevice(dxgi_context.adapter.get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&ctx.dev));

	if (FAILED(hr))
		return {};

	D3D12_COMMAND_QUEUE_DESC queue_desc = {};
	queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
	if (FAILED(ctx.dev->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&ctx.queue))))
		return {};

	for (unsigned i = 0; i < 2; i++)
	{
		if (FAILED(ctx.dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&ctx.allocator[i]))))
			return {};
		if (FAILED(ctx.dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
		                                      ctx.allocator[i].get(), nullptr,
		                                      IID_PPV_ARGS(&ctx.list[i]))))
			return {};
		ctx.list[i]->Close();
	}

	DXGI_ADAPTER_DESC desc;
	dxgi_context.adapter->GetDesc(&desc);
	ctx.luid = desc.AdapterLuid;

	if (FAILED(hr))
	{
		LOGE("Failed to create D3D12 device.\n");
		return {};
	}

	ctx.dxgi = dxgi_context;
	return ctx;
}

static bool init_swapchain(SDL_Window *window, D3DContext &ctx)
{
	SDL_PropertiesID props = SDL_GetWindowProperties(window);
	SDL_LockProperties(props);
	HWND hwnd = static_cast<HWND>(SDL_GetPointerProperty(props, "SDL.window.win32.hwnd", nullptr));
	SDL_UnlockProperties(props);

	DXGI_SWAP_CHAIN_DESC desc = {};
	desc.BufferCount = 2;
	desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	desc.OutputWindow = hwnd;
	desc.Windowed = TRUE;
	desc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	desc.BufferDesc.Width = Width;
	desc.BufferDesc.Height = Height;
	desc.BufferDesc.Scaling = DXGI_MODE_SCALING_STRETCHED;
	desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	desc.SampleDesc.Count = 1;

	ComPtr<IDXGISwapChain> swapchain;

	auto hr = ctx.dxgi.factory->CreateSwapChain(ctx.queue.get(), &desc, &swapchain);
	if (FAILED(hr))
	{
		LOGE("Failed to create swapchain.\n");
		return false;
	}

	if (FAILED(swapchain->QueryInterface(&ctx.swapchain)))
		return false;

	for (unsigned i = 0; i < 2; i++)
	{
		if (FAILED(ctx.swapchain->GetBuffer(i, IID_PPV_ARGS(&ctx.back_buffers[i]))))
			return false;
	}

	return true;
}

int main()
{
	if (!SDL_Init(SDL_INIT_VIDEO))
		return EXIT_FAILURE;

	Granite::Global::init(Granite::Global::MANAGER_FEATURE_DEFAULT_BITS, 1);
	Granite::Filesystem::setup_default_filesystem(GRANITE_FILESYSTEM(), ASSET_DIRECTORY);

	auto ctx = create_d3d12_device();
	if (!ctx.dev)
		return EXIT_FAILURE;

	SDL_Window *window = SDL_CreateWindow("D3D12 interop", Width, Height, 0);
	if (!window)
	{
		LOGE("Failed to create window.\n");
		return EXIT_FAILURE;
	}

	if (!init_swapchain(window, ctx))
		return EXIT_FAILURE;

	if (!Context::init_loader(nullptr))
		return EXIT_FAILURE;

	Context vk;
	Device device;
	Context::SystemHandles handles = {};
	handles.filesystem = GRANITE_FILESYSTEM();
	vk.set_system_handles(handles);
	if (!vk.init_instance_and_device(nullptr, 0, nullptr, 0))
	{
		LOGE("Failed to create Vulkan device.\n");
		return EXIT_FAILURE;
	}

	device.set_context(vk);

	if (!device.get_device_features().supports_external)
	{
		LOGE("Vulkan device does not support external.\n");
		return EXIT_FAILURE;
	}

	if (memcmp(device.get_device_features().vk11_props.deviceLUID,
	           &ctx.luid, VK_LUID_SIZE) != 0)
	{
		LOGE("LUID mismatch.\n");
		return EXIT_FAILURE;
	}

	D3D12_RESOURCE_DESC desc = {};
	desc.SampleDesc.Count = 1;
	desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
	desc.Width = Width;
	desc.Height = Height;
	desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
	desc.MipLevels = 1;
	desc.DepthOrArraySize = 1;
	desc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
	desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;

	D3D12_HEAP_PROPERTIES heap_props = {};
	heap_props.Type = D3D12_HEAP_TYPE_DEFAULT;
	if (FAILED(ctx.dev->CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_SHARED,
	                                            &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&ctx.texture))))
	{
		LOGE("Failed to create texture.\n");
		return EXIT_FAILURE;
	}

	desc.Format = DXGI_FORMAT_NV12;
	if (FAILED(ctx.dev->CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_SHARED,
		&desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&ctx.texture_nv12))))
	{
		LOGE("Failed to create texture.\n");
		return EXIT_FAILURE;
	}

	ExternalHandle imported_image;
	if (FAILED(ctx.dev->CreateSharedHandle(ctx.texture.get(), nullptr, GENERIC_ALL, nullptr, &imported_image.handle)))
		return EXIT_FAILURE;
	imported_image.memory_handle_type = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;

	auto image_info = ImageCreateInfo::render_target(Width, Height, VK_FORMAT_R8G8B8A8_UNORM);
	image_info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
	image_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

	if (device.get_device_features().vk12_props.driverID == VK_DRIVER_ID_NVIDIA_PROPRIETARY)
	{
		// NV workaround, otherwise it doesn't match swizzle in D3D.
		image_info.usage |= VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR;
	}

	image_info.misc = IMAGE_MISC_EXTERNAL_MEMORY_BIT;
	image_info.external = imported_image;
	auto image = device.create_image(image_info);

	if (FAILED(ctx.dev->CreateSharedHandle(ctx.texture_nv12.get(), nullptr, GENERIC_ALL, nullptr, &imported_image.handle)))
		return EXIT_FAILURE;
	imported_image.memory_handle_type = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;
	image_info.format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
	image_info.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT; // Needed to take per-plane views.
	auto image_nv12 = device.create_image(image_info);

	if (!image)
	{
		LOGE("Failed to create image.\n");
		return EXIT_FAILURE;
	}

	if (!image_nv12)
	{
		LOGE("Failed to create image.\n");
		return EXIT_FAILURE;
	}

	// Create decode images.
	image_info.format = VK_FORMAT_R8_UNORM;
	image_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
	image_info.misc = 0;
	image_info.external = {};
	auto image_decode_y = device.create_image(image_info);

	image_info.width >>= 1;
	image_info.height >>= 1;
	image_info.format = VK_FORMAT_R8_UNORM;
	image_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
	image_info.misc = 0;
	image_info.external = {};
	auto image_decode_cb = device.create_image(image_info);
	auto image_decode_cr = device.create_image(image_info);

	if (!image_decode_y || !image_decode_cb || !image_decode_cr)
	{
		LOGE("Failed to create images.\n");
		return EXIT_FAILURE;
	}

	ComPtr<ID3D12Resource> staging_buffer;
	{
		auto mapping = GRANITE_FILESYSTEM()->open_readonly_mapping("assets://test.yuv");
		if (!mapping)
		{
			LOGE("Failed to create mapping.\n");
			return EXIT_FAILURE;
		}

		D3D12_RESOURCE_DESC desc = {};
		desc.Width = mapping->get_size();
		desc.Height = 1;
		desc.DepthOrArraySize = 1;
		desc.MipLevels = 1;
		desc.SampleDesc.Count = 1;
		desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
		desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

		D3D12_HEAP_PROPERTIES heap_props = {};
		heap_props.Type = D3D12_HEAP_TYPE_UPLOAD;
		
		if (FAILED(ctx.dev->CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_CREATE_NOT_ZEROED, &desc,
			D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&staging_buffer))))
		{
			LOGE("Failed to create buffer.\n");
			return EXIT_FAILURE;
		}

		void *ptr = nullptr;
		staging_buffer->Map(0, nullptr, &ptr);
		memcpy(ptr, mapping->data(), mapping->get_size());
		staging_buffer->Unmap(0, nullptr);
	}

	ComPtr<ID3D12DescriptorHeap> rtv_heap;
	D3D12_DESCRIPTOR_HEAP_DESC heap_desc = {};
	heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
	heap_desc.NumDescriptors = 2;
	if (FAILED(ctx.dev->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&rtv_heap))))
	{
		LOGE("Failed to create RTV heap.\n");
		return EXIT_FAILURE;
	}

	for (int i = 0; i < 2; i++)
	{
		D3D12_RENDER_TARGET_VIEW_DESC rtv_desc = {};
		rtv_desc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
		rtv_desc.Format = DXGI_FORMAT_R8_UNORM;
		rtv_desc.Texture2D.PlaneSlice = 0;

		auto heap_handle = rtv_heap->GetCPUDescriptorHandleForHeapStart();
		ctx.dev->CreateRenderTargetView(ctx.texture_nv12.get(), &rtv_desc, heap_handle);
		rtv_desc.Format = DXGI_FORMAT_R8G8_UNORM;
		rtv_desc.Texture2D.PlaneSlice = 1;
		heap_handle.ptr += ctx.dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
		ctx.dev->CreateRenderTargetView(ctx.texture_nv12.get(), &rtv_desc, heap_handle);
	}

	if (FAILED(ctx.dev->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&ctx.fence))))
	{
		LOGE("Failed to create fence.\n");
		return EXIT_FAILURE;
	}

	auto timeline = device.request_semaphore_external(VK_SEMAPHORE_TYPE_TIMELINE,
	                                                  VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT);
	if (!timeline)
	{
		LOGE("Failed to create timeline.\n");
		return EXIT_FAILURE;
	}

	ExternalHandle fence_handle;
	fence_handle.semaphore_handle_type = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT;
	if (FAILED(ctx.dev->CreateSharedHandle(ctx.fence.get(), nullptr, GENERIC_ALL, nullptr, &fence_handle.handle)))
	{
		LOGE("Failed to create shared fence handle.\n");
		return EXIT_FAILURE;
	}

	if (!timeline->import_from_handle(fence_handle))
	{
		LOGE("Failed to import timeline.\n");
		return EXIT_FAILURE;
	}

	PyroWave::Encoder encoder;
	PyroWave::Decoder decoder;

	if (!encoder.init(&device, Width, Height, PyroWave::ChromaSubsampling::Chroma420))
	{
		LOGE("Failed to init encoder.\n");
		return EXIT_FAILURE;
	}

	if (!decoder.init(&device, Width, Height, PyroWave::ChromaSubsampling::Chroma420))
	{
		LOGE("Failed to init decoder.\n");
		return EXIT_FAILURE;
	}

	uint64_t timeline_value = 0;
	unsigned frame_count = 0;
	unsigned wait_context;

	bool alive = true;
	SDL_Event e;
	while (alive)
	{
		while (SDL_PollEvent(&e))
			if (e.type == SDL_EVENT_QUIT)
				alive = false;

		wait_context = frame_count % 2;
		auto *allocator = ctx.allocator[wait_context].get();
		auto *list = ctx.list[wait_context].get();

		// Render dummy NV12 in D3D12
		{
			ctx.fence->SetEventOnCompletion(ctx.wait_timeline[wait_context], nullptr);
			allocator->Reset();
			list->Reset(allocator, nullptr);

			D3D12_RESOURCE_BARRIER barrier = {};
			barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
			barrier.Transition.pResource = ctx.texture_nv12.get();
			barrier.Transition.Subresource = -1;
			barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
			barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
			list->ResourceBarrier(1, &barrier);

			D3D12_BOX srcBox = {};
			srcBox.right = Width;
			srcBox.bottom = Height;
			srcBox.back = 1;
			D3D12_TEXTURE_COPY_LOCATION dstY = {}, srcY = {};

			dstY.pResource = ctx.texture_nv12.get();
			dstY.SubresourceIndex = 0;
			dstY.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;

			srcY.pResource = staging_buffer.get();
			srcY.PlacedFootprint.Footprint.Width = Width;
			srcY.PlacedFootprint.Footprint.Height = Height;
			srcY.PlacedFootprint.Footprint.RowPitch = Width;
			srcY.PlacedFootprint.Footprint.Depth = 1;
			srcY.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8_UNORM;
			srcY.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
			list->CopyTextureRegion(&dstY, 0, 0, 0, &srcY, &srcBox);

			dstY.SubresourceIndex = 1;
			srcY.PlacedFootprint.Footprint.Width = Width >> 1;
			srcY.PlacedFootprint.Footprint.Height = Height >> 1;
			srcY.PlacedFootprint.Footprint.RowPitch = Width;
			srcY.PlacedFootprint.Footprint.Depth = 1;
			srcY.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8_UNORM;
			srcY.PlacedFootprint.Offset = Width * Height;
			srcBox.right = Width >> 1;
			srcBox.bottom = Height >> 1;
			srcBox.back = 1;
			list->CopyTextureRegion(&dstY, 0, 0, 0, &srcY, &srcBox);

			std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
			list->ResourceBarrier(1, &barrier);

			// Submit and signal fence.
			list->Close();
			ID3D12CommandList *submit_list = list;
			ctx.queue->ExecuteCommandLists(1, &submit_list);
			timeline_value++;
			ctx.queue->Signal(ctx.fence.get(), timeline_value);
			ctx.wait_timeline[wait_context] = timeline_value;

			// Make Vulkan wait on NV12 rendering to complete.
			auto waiter = device.request_timeline_semaphore_as_binary(*timeline, timeline_value);
			waiter->signal_external();
			device.add_wait_semaphore(CommandBuffer::Type::Generic, std::move(waiter),
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, true);
		}

		// Encode + Decode in PyroWave
		{
			auto cmd = device.request_command_buffer();
			cmd->acquire_image_barrier(*image_nv12,
				VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

			ImageViewCreateInfo view_info = {};
			view_info.aspect = VK_IMAGE_ASPECT_PLANE_0_BIT;
			view_info.image = image_nv12.get();
			view_info.format = VK_FORMAT_R8_UNORM;
			view_info.layers = 1;
			view_info.levels = 1;
			view_info.view_type = VK_IMAGE_VIEW_TYPE_2D;

			// PyroWave processes the planes separately, but we can just use image views to separate out NV12 to yuv420p.
			auto Y_view = device.create_image_view(view_info);
			view_info.format = VK_FORMAT_R8G8_UNORM;
			view_info.aspect = VK_IMAGE_ASPECT_PLANE_1_BIT;
			view_info.swizzle.r = VK_COMPONENT_SWIZZLE_R;
			auto Cb_view = device.create_image_view(view_info);
			view_info.swizzle.r = VK_COMPONENT_SWIZZLE_G;
			auto Cr_view = device.create_image_view(view_info);

			PyroWave::ViewBuffers views = {{ Y_view.get(), Cb_view.get(), Cr_view.get() }};

			// These buffers should be device-local and copy over to host after the fact, but this is for simplicity.
			BufferCreateInfo bufinfo = {};
			bufinfo.domain = BufferDomain::CachedHost;
			bufinfo.size = encoder.get_meta_required_size();
			bufinfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
			auto meta_buffer = device.create_buffer(bufinfo);

			constexpr VkDeviceSize TargetSize = 400000;
			bufinfo.size = TargetSize + encoder.get_meta_required_size();
			auto bitstream_buffer = device.create_buffer(bufinfo);

			PyroWave::Encoder::BitstreamBuffers bitstream = {};
			bitstream.meta = { meta_buffer.get(), 0, meta_buffer->get_create_info().size };
			bitstream.bitstream = { bitstream_buffer.get(), 0, bitstream_buffer->get_create_info().size };
			bitstream.target_size = TargetSize;
			if (!encoder.encode(*cmd, views, bitstream))
			{
				LOGE("Failed to encode.\n");
				return EXIT_FAILURE;
			}

			cmd->release_image_barrier(*image_nv12,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0);

			cmd->barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
			Fence fence;
			device.submit(cmd, &fence);
			fence->wait();

			auto *mapped_meta = device.map_host_buffer(*meta_buffer, MEMORY_ACCESS_READ_BIT);
			auto *mapped_bitstream = device.map_host_buffer(*bitstream_buffer, MEMORY_ACCESS_READ_BIT);

			// For networking purposes, can select split point.
			if (encoder.compute_num_packets(mapped_meta, TargetSize) != 1)
			{
				LOGE("Expected that we could fit 100k into one big packet.\n");
				return EXIT_FAILURE;
			}

			std::vector<uint8_t> bitstream_data(TargetSize);
			PyroWave::Encoder::Packet packet;
			if (!encoder.packetize(&packet, TargetSize, bitstream_data.data(), bitstream_data.size(), mapped_meta, mapped_bitstream))
			{
				LOGE("Failed to packetize.\n");
				return EXIT_FAILURE;
			}

			// Push packet to decoder.
			decoder.push_packet(bitstream_data.data() + packet.offset, packet.size);
			if (!decoder.decode_is_ready(false))
			{
				LOGE("Decoding should be ready now ...\n");
				return EXIT_FAILURE;
			}

			// Decode to YCbCr.
			cmd = device.request_command_buffer();
			cmd->image_barrier(*image_decode_y,
				VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
			cmd->image_barrier(*image_decode_cb,
				VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
			cmd->image_barrier(*image_decode_cr,
				VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

			views = {{ &image_decode_y->get_view(), &image_decode_cb->get_view(), &image_decode_cr->get_view() }};
			if (!decoder.decode(*cmd, views))
			{
				LOGE("Failed to decode.\n");
				return EXIT_FAILURE;
			}

			cmd->image_barrier(*image_decode_y, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
				VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
			cmd->image_barrier(*image_decode_cb, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
				VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
			cmd->image_barrier(*image_decode_cr, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
				VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

			// YCbCr -> RGB.

			RenderPassInfo rp_info;
			rp_info.num_color_attachments = 1;
			rp_info.color_attachments[0] = &image->get_view();
			rp_info.store_attachments = 1u << 0;
			rp_info.clear_attachments = 1u << 0;

			// Don't need to reacquire from external queue family if we don't care about the contents being preserved.
			cmd->image_barrier(*image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			                   VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
			                   VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);

			cmd->begin_render_pass(rp_info);

			cmd->set_opaque_sprite_state();
			cmd->set_program("assets://quad.vert", "assets://quad.frag");
			cmd->set_texture(0, 0, image_decode_y->get_view());
			cmd->set_texture(0, 1, image_decode_cb->get_view());
			cmd->set_texture(0, 2, image_decode_cr->get_view());
			cmd->set_sampler(0, 3, StockSampler::LinearClamp);
			cmd->draw(3);

			// For non-Vulkan compatible APIs we have to use GENERAL layout.
			cmd->end_render_pass();
			cmd->release_image_barrier(*image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
				VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_QUEUE_FAMILY_EXTERNAL);
			device.submit(cmd);
		}

		// Signal ID3D12Fence and wait on it in D3D12.
		{
			timeline_value++;
			auto signal = device.request_timeline_semaphore_as_binary(*timeline, timeline_value);
			device.submit_empty(CommandBuffer::Type::Generic, nullptr, signal.get());
			ctx.queue->Wait(ctx.fence.get(), timeline_value);
		}

		unsigned swap_index = ctx.swapchain->GetCurrentBackBufferIndex();

		// Blit shared texture to back buffer.
		{
			D3D12_BOX box = {};
			box.back = 1;
			box.right = Width;
			box.bottom = Height;

			D3D12_TEXTURE_COPY_LOCATION dst = {};
			D3D12_TEXTURE_COPY_LOCATION src = {};
			dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
			dst.pResource = ctx.back_buffers[swap_index].get();
			src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
			src.pResource = ctx.texture.get();

			list->Reset(allocator, nullptr);
			list->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);
			list->Close();

			ID3D12CommandList *submit_list = list;
			ctx.queue->ExecuteCommandLists(1, &submit_list);
		}

		// Release the texture to Vulkan.
		{
			timeline_value++;
			ctx.queue->Signal(ctx.fence.get(), timeline_value);
			ctx.wait_timeline[wait_context] = timeline_value;

			auto waiter = device.request_timeline_semaphore_as_binary(*timeline, timeline_value);
			waiter->signal_external();
			device.add_wait_semaphore(CommandBuffer::Type::Generic, std::move(waiter),
			                          VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, true);
		}

		ctx.swapchain->Present(1, 0);
		device.next_frame_context();
		frame_count++;
	}

	ctx.fence->SetEventOnCompletion(timeline_value, nullptr);
	ctx = {};
	SDL_DestroyWindow(window);
	SDL_Quit();
}
