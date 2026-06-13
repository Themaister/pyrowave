// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

#include "vulkan/vulkan.h"
#include "pyrowave.h"
#include <stdio.h>
#include <cstdlib>
#include <exception>
#include <vector>

// Smoke test the C API.

#define ASSERT_THAT(x) do { \
	if (!(x)) { fprintf(stderr, "Fatal error executing %s at line %d.\n", #x, __LINE__); std::terminate(); } \
} while(false)

#define CHECKED(x) do { \
	pyrowave_result res = x; \
	if (res != PYROWAVE_SUCCESS) { fprintf(stderr, "Got pyrowave result %d while executing %s at line %d.\n", res, #x, __LINE__); std::terminate(); } \
} while(false)

static void test_encoder_create_validation()
{
	pyrowave_encoder_create_info info = {};
	info.width = 64;
	info.height = 64;

	// No device.
	pyrowave_encoder encoder, dummy;
	ASSERT_THAT(pyrowave_encoder_create(&info, &encoder) == PYROWAVE_ERROR_INVALID_ARGUMENT);

	pyrowave_device device;
	CHECKED(pyrowave_create_default_device(&device));

	info.device = device;
	CHECKED(pyrowave_encoder_create(&info, &encoder));

	// 0 size not allowed.
	info.width = 0;
	info.height = 0;
	ASSERT_THAT(pyrowave_encoder_create(&info, &dummy) == PYROWAVE_ERROR_INVALID_ARGUMENT);

	// Odd size not allowed.
	info.width = 65;
	info.height = 64;
	ASSERT_THAT(pyrowave_encoder_create(&info, &dummy) == PYROWAVE_ERROR_INVALID_ARGUMENT);

	info.width = 64;
	info.height = 65;
	ASSERT_THAT(pyrowave_encoder_create(&info, &dummy) == PYROWAVE_ERROR_INVALID_ARGUMENT);

	// Odd size allowed for 444.
	info.chroma = PYROWAVE_CHROMA_SUBSAMPLING_444;
	CHECKED(pyrowave_encoder_create(&info, &dummy));
	pyrowave_encoder_destroy(dummy);

	pyrowave_encoder_destroy(encoder);
	pyrowave_device_destroy(device);
}

static void test_decoder_create_validation()
{
	pyrowave_decoder_create_info info = {};
	info.width = 64;
	info.height = 64;

	// No device.
	pyrowave_decoder decoder, dummy;
	ASSERT_THAT(pyrowave_decoder_create(&info, &decoder) == PYROWAVE_ERROR_INVALID_ARGUMENT);

	pyrowave_device device;
	CHECKED(pyrowave_create_default_device(&device));

	info.device = device;
	CHECKED(pyrowave_decoder_create(&info, &decoder));

	// 0 size not allowed.
	info.width = 0;
	info.height = 0;
	ASSERT_THAT(pyrowave_decoder_create(&info, &dummy) == PYROWAVE_ERROR_INVALID_ARGUMENT);

	// Odd size not allowed.
	info.width = 65;
	info.height = 64;
	ASSERT_THAT(pyrowave_decoder_create(&info, &dummy) == PYROWAVE_ERROR_INVALID_ARGUMENT);

	info.width = 64;
	info.height = 65;
	ASSERT_THAT(pyrowave_decoder_create(&info, &dummy) == PYROWAVE_ERROR_INVALID_ARGUMENT);

	// Odd size allowed for 444.
	info.chroma = PYROWAVE_CHROMA_SUBSAMPLING_444;
	CHECKED(pyrowave_decoder_create(&info, &dummy));
	pyrowave_decoder_destroy(dummy);

	// Smoke test that creating device on fragment path doesn't explode.
	info.fragment_path = true;
	CHECKED(pyrowave_decoder_create(&info, &dummy));
	pyrowave_decoder_destroy(dummy);

	pyrowave_decoder_destroy(decoder);
	pyrowave_device_destroy(device);
}

static void test_decode_cpu_buffer_validation(bool fragment_path)
{
	pyrowave_decoder_create_info info = {};
	info.width = 16;
	info.height = 16;
	info.fragment_path = fragment_path;

	pyrowave_decoder decoder;
	CHECKED(pyrowave_create_default_device(&info.device));
	CHECKED(pyrowave_decoder_create(&info, &decoder));

	// This shouldn't be ready.
	ASSERT_THAT(!pyrowave_decoder_decode_is_ready(decoder, false));
	ASSERT_THAT(!pyrowave_decoder_decode_is_ready(decoder, true));

	pyrowave_cpu_buffer cpu_buffer = {};

	uint8_t luma[16][16] = {};
	uint8_t cb[8][16] = {}; // Test strided readback while we're at it.
	uint8_t cr[8][16] = {};

	cpu_buffer.format = PYROWAVE_CPU_BUFFER_FORMAT_YUV420P;
	cpu_buffer.row_stride_in_bytes[0] = 16;
	cpu_buffer.row_stride_in_bytes[1] = 16;
	cpu_buffer.row_stride_in_bytes[2] = 16;
	cpu_buffer.plane_size_in_bytes[0] = 16 * 16;
	cpu_buffer.plane_size_in_bytes[1] = 16 * 8;
	cpu_buffer.plane_size_in_bytes[2] = 16 * 8;
	cpu_buffer.data[0] = &luma[0][0];
	cpu_buffer.data[1] = &cb[0][0];
	cpu_buffer.data[2] = &cr[0][0];

	cpu_buffer.width = 16;
	cpu_buffer.height = 16;

	// NV12 is banned for decode.
	cpu_buffer.format = PYROWAVE_CPU_BUFFER_FORMAT_NV12;
	ASSERT_THAT(pyrowave_decoder_decode_cpu_buffer_synchronous(decoder, &cpu_buffer) == PYROWAVE_ERROR_INVALID_ARGUMENT);

	cpu_buffer.format = PYROWAVE_CPU_BUFFER_FORMAT_YUV420P;
	CHECKED(pyrowave_decoder_decode_cpu_buffer_synchronous(decoder, &cpu_buffer));

	// Assert that we do indeed decode a gray image.
	for (uint32_t y = 0; y < 16; y++)
		for (uint32_t x = 0; x < 16; x++)
			ASSERT_THAT(luma[y][x] == 0x7f || luma[y][x] == 0x80);

	for (uint32_t y = 0; y < 8; y++)
		for (uint32_t x = 0; x < 8; x++)
			ASSERT_THAT(cb[y][x] == 0x7f || cb[y][x] == 0x80);

	for (uint32_t y = 0; y < 8; y++)
		for (uint32_t x = 0; x < 8; x++)
			ASSERT_THAT(cr[y][x] == 0x7f || cr[y][x] == 0x80);

	pyrowave_decoder_destroy(decoder);
	pyrowave_device_destroy(info.device);
}

static void test_encode_cpu_buffer_validation(bool nv12)
{
	pyrowave_encoder_create_info info = {};
	info.width = 16;
	info.height = 16;

	pyrowave_encoder encoder;
	CHECKED(pyrowave_create_default_device(&info.device));
	CHECKED(pyrowave_encoder_create(&info, &encoder));

	const pyrowave_rate_control rate_control = { 1024 };
	pyrowave_cpu_buffer cpu_buffer = {};

	uint8_t y[16][16] = {};
	uint8_t cb[8][8] = {};
	uint8_t cr[8][8] = {};

	cpu_buffer.format = nv12 ? PYROWAVE_CPU_BUFFER_FORMAT_NV12 : PYROWAVE_CPU_BUFFER_FORMAT_YUV420P;
	cpu_buffer.row_stride_in_bytes[0] = 16;
	cpu_buffer.row_stride_in_bytes[1] = nv12 ? 16 : 8;
	cpu_buffer.row_stride_in_bytes[2] = nv12 ? 0 : 8;
	cpu_buffer.plane_size_in_bytes[0] = 16 * 16;
	cpu_buffer.plane_size_in_bytes[1] = (nv12 ? 2 : 1) * 8 * 8;
	cpu_buffer.plane_size_in_bytes[2] = nv12 ? 0 : 8 * 8;
	cpu_buffer.data[0] = &y[0][0];
	cpu_buffer.data[1] = &cb[0][0];
	cpu_buffer.data[2] = &cr[0][0];

	cpu_buffer.width = 16;
	cpu_buffer.height = 16;
	CHECKED(pyrowave_encoder_encode_cpu_synchronous(encoder, &cpu_buffer, &rate_control));

	// Mismatching width/height against encoder.
	cpu_buffer.width = 15;
	cpu_buffer.height = 16;
	ASSERT_THAT(pyrowave_encoder_encode_cpu_synchronous(encoder, &cpu_buffer, &rate_control) == PYROWAVE_ERROR_INVALID_ARGUMENT);

	cpu_buffer.width = 16;
	cpu_buffer.height = 15;
	ASSERT_THAT(pyrowave_encoder_encode_cpu_synchronous(encoder, &cpu_buffer, &rate_control) == PYROWAVE_ERROR_INVALID_ARGUMENT);

	// Too small row strides.
	cpu_buffer.width = 16;
	cpu_buffer.height = 16;
	cpu_buffer.row_stride_in_bytes[1] = nv12 ? 15 : 7;
	ASSERT_THAT(pyrowave_encoder_encode_cpu_synchronous(encoder, &cpu_buffer, &rate_control) == PYROWAVE_ERROR_INVALID_ARGUMENT);

	// Too small plane size.
	cpu_buffer.row_stride_in_bytes[1] = nv12 ? 16 : 8;
	cpu_buffer.plane_size_in_bytes[1] = (nv12 ? 2 : 1) * 8 * 8 - 1;
	ASSERT_THAT(pyrowave_encoder_encode_cpu_synchronous(encoder, &cpu_buffer, &rate_control) == PYROWAVE_ERROR_INVALID_ARGUMENT);

	pyrowave_encoder_destroy(encoder);
	pyrowave_device_destroy(info.device);
}

static void test_basic_encoder_roundtrip(bool fragment_decode, bool nv12_encode, pyrowave_chroma_subsampling chroma)
{
	if (chroma == PYROWAVE_CHROMA_SUBSAMPLING_444 && nv12_encode)
		return;

	if (fragment_decode)
		return;

	pyrowave_device device;
	CHECKED(pyrowave_create_default_device(&device));

	constexpr int Width = 34;
	constexpr int Height = 30;

	pyrowave_decoder_create_info decoder_info = {};
	decoder_info.device = device;
	decoder_info.width = Width; // Test somewhat odd size. Quite relevant for fragment path as well.
	decoder_info.height = Height;
	decoder_info.fragment_path = fragment_decode;
	decoder_info.chroma = chroma;

	pyrowave_encoder_create_info encoder_info = {};
	encoder_info.device = device;
	encoder_info.width = Width;
	encoder_info.height = Height;
	encoder_info.chroma = chroma;

	pyrowave_decoder decoder;
	pyrowave_encoder encoder;
	CHECKED(pyrowave_decoder_create(&decoder_info, &decoder));
	CHECKED(pyrowave_encoder_create(&encoder_info, &encoder));

	uint8_t luma[Height][Width] = {};
	uint8_t cb[Height][Width] = {};
	uint8_t cr[Height][Width] = {};
	uint8_t cbcr[Height][Width][2] = {};

	uint8_t decode_luma[Height][Width] = {};
	uint8_t decode_cb[Height][Width] = {};
	uint8_t decode_cr[Height][Width] = {};

	pyrowave_cpu_buffer cpu_buffer = {};

	cpu_buffer.format =
			nv12_encode
				? PYROWAVE_CPU_BUFFER_FORMAT_NV12
				: (chroma == PYROWAVE_CHROMA_SUBSAMPLING_444
					   ? PYROWAVE_CPU_BUFFER_FORMAT_YUV444P
					   : PYROWAVE_CPU_BUFFER_FORMAT_YUV420P);

	cpu_buffer.row_stride_in_bytes[0] = Width;
	cpu_buffer.row_stride_in_bytes[1] = nv12_encode ? sizeof(cbcr[0]) : sizeof(cb[0]);
	cpu_buffer.row_stride_in_bytes[2] = nv12_encode ? 0 : sizeof(cr[0]);
	cpu_buffer.plane_size_in_bytes[0] = sizeof(luma);
	cpu_buffer.plane_size_in_bytes[1] = nv12_encode ? sizeof(cbcr) : sizeof(cb);
	cpu_buffer.plane_size_in_bytes[2] = nv12_encode ? 0 : sizeof(cr);
	cpu_buffer.data[0] = &luma[0][0];

	if (nv12_encode)
	{
		cpu_buffer.data[1] = &cbcr[0][0][0];
	}
	else
	{
		cpu_buffer.data[1] = &cb[0][0];
		cpu_buffer.data[2] = &cr[0][0];
	}

	for (int y = 0; y < Height; y++)
	{
		for (int x = 0; x < Width; x++)
		{
			luma[y][x] = uint8_t(3 * x + 5 * y);

			uint8_t cb_signal = 7 * x + 3 * y;
			uint8_t cr_signal = 3 * x + 5 * y;

			if (nv12_encode)
			{
				cbcr[y][x][0] = cb_signal;
				cbcr[y][x][1] = cr_signal;
			}
			else
			{
				cb[y][x] = cb_signal;
				cr[y][x] = cr_signal;
			}
		}
	}

	cpu_buffer.width = Width;
	cpu_buffer.height = Height;
	const pyrowave_rate_control rate_control = { 64 * 1024 }; // Just give it something massive.
	CHECKED(pyrowave_encoder_encode_cpu_synchronous(encoder, &cpu_buffer, &rate_control));

	size_t num_packets;
	CHECKED(pyrowave_encoder_compute_num_packets(encoder, 64 * 1024, &num_packets));
	ASSERT_THAT(num_packets == 1);

	std::vector<uint8_t> bitstream(64 * 1024);
	pyrowave_packet packet = {};
	CHECKED(pyrowave_encoder_packetize(encoder, &packet, 64 * 1024, &num_packets, bitstream.data(), bitstream.size()));
	ASSERT_THAT(num_packets == 1);
	ASSERT_THAT(packet.offset == 0);
	ASSERT_THAT(packet.size != 0);
	ASSERT_THAT(packet.size <= bitstream.size());
	bitstream.resize(packet.size);

	CHECKED(pyrowave_decoder_push_packet(decoder, bitstream.data() + packet.offset, packet.size));
	ASSERT_THAT(pyrowave_decoder_decode_is_ready(decoder, false));
	pyrowave_decoder_clear(decoder);
	ASSERT_THAT(!pyrowave_decoder_decode_is_ready(decoder, false));
	CHECKED(pyrowave_decoder_push_packet(decoder, bitstream.data() + packet.offset, packet.size));
	ASSERT_THAT(pyrowave_decoder_decode_is_ready(decoder, false));

	cpu_buffer.data[0] = &decode_luma[0][0];
	cpu_buffer.data[1] = &decode_cb[0][0];
	cpu_buffer.data[2] = &decode_cr[0][0];
	cpu_buffer.row_stride_in_bytes[1] = sizeof(decode_cb[0]);
	cpu_buffer.row_stride_in_bytes[2] = sizeof(decode_cr[0]);
	cpu_buffer.plane_size_in_bytes[1] = sizeof(decode_cb);
	cpu_buffer.plane_size_in_bytes[2] = sizeof(decode_cr);
	cpu_buffer.format = chroma == PYROWAVE_CHROMA_SUBSAMPLING_444 ?
			PYROWAVE_CPU_BUFFER_FORMAT_YUV444P : PYROWAVE_CPU_BUFFER_FORMAT_YUV420P;

	CHECKED(pyrowave_decoder_decode_cpu_buffer_synchronous(decoder, &cpu_buffer));

	for (int y = 0; y < Height; y++)
	{
		for (int x = 0; x < Width; x++)
		{
			int d = std::abs(int(decode_luma[y][x]) - int(luma[y][x]));
			// With the "infinite" bitrate we get here,
			// accept a maximum 1 ULP error.
			ASSERT_THAT(d <= 1);

			if (chroma == PYROWAVE_CHROMA_SUBSAMPLING_444 || (!nv12_encode && y < Height / 2 && x < Width / 2))
			{
				// Allow more error for chroma.
				d = std::abs(int(decode_cb[y][x]) - int(cb[y][x]));
				ASSERT_THAT(d <= 1);
				d = std::abs(int(decode_cr[y][x]) - int(cr[y][x]));
				ASSERT_THAT(d <= 1);
			}
		}
	}

	if (nv12_encode)
	{
		for (int y = 0; y < Height / 2; y++)
		{
			for (int x = 0; x < Width / 2; x++)
			{
				int d = std::abs(int(decode_cb[y][x]) - int(cbcr[y][x][0]));
				ASSERT_THAT(d <= 1);
				d = std::abs(int(decode_cr[y][x]) - int(cbcr[y][x][1]));
				ASSERT_THAT(d <= 1);
			}
		}
	}

	pyrowave_decoder_destroy(decoder);
	pyrowave_encoder_destroy(encoder);
	pyrowave_device_destroy(device);
}

int main()
{
	for (int variant = 0; variant < 8; variant++)
	{
		test_basic_encoder_roundtrip(
			(variant & 1) != 0, (variant & 2) != 0,
			(variant & 4) != 0 ? PYROWAVE_CHROMA_SUBSAMPLING_444 : PYROWAVE_CHROMA_SUBSAMPLING_420);
	}

	test_decode_cpu_buffer_validation(false);
	test_decode_cpu_buffer_validation(true);
	test_encode_cpu_buffer_validation(false);
	test_encode_cpu_buffer_validation(true);
	test_encoder_create_validation();
	test_decoder_create_validation();
	printf("Passed all tests :)\n");
}
