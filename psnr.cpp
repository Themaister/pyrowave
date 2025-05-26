// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

#include "yuv4mpeg.hpp"
#include <stdint.h>
#include <cmath>

int main(int argc, char **argv)
{
	if (argc != 3)
	{
		fprintf(stderr, "Usage: a.y4m b.y4m\n");
		return EXIT_FAILURE;
	}

	YUV4MPEGFile a, b;

	if (!a.open_read(argv[1]))
	{
		fprintf(stderr, "Failed to open %s.\n", argv[1]);
		return EXIT_FAILURE;
	}

	if (!b.open_read(argv[2]))
	{
		fprintf(stderr, "Failed to open %s.\n", argv[2]);
		return EXIT_FAILURE;
	}

	if (a.get_width() != b.get_width() || a.get_height() != b.get_height())
	{
		fprintf(stderr, "Mismatch in parameters (%d, %d) != (%d, %d)\n",
				a.get_width(), a.get_height(), b.get_width(), b.get_height());
		return EXIT_FAILURE;
	}

	int num_luma_pixels = a.get_width() * a.get_height();
	int num_chroma_pixels = (a.get_width() / 2) * (a.get_height() / 2);
	std::unique_ptr<uint8_t[]> Y[2] = {
			std::unique_ptr<uint8_t[]>(new uint8_t[num_luma_pixels]),
			std::unique_ptr<uint8_t[]>(new uint8_t[num_luma_pixels]) };

	std::unique_ptr<uint8_t[]> Cb[2] = {
			std::unique_ptr<uint8_t[]>(new uint8_t[num_chroma_pixels]),
			std::unique_ptr<uint8_t[]>(new uint8_t[num_chroma_pixels]) };

	std::unique_ptr<uint8_t[]> Cr[2] = {
			std::unique_ptr<uint8_t[]>(new uint8_t[num_chroma_pixels]),
			std::unique_ptr<uint8_t[]>(new uint8_t[num_chroma_pixels]) };

	uint64_t total_peak_signal[3] = {};
	uint64_t total_error[3] = {};
	uint64_t frame_peak_signal[3] = {};
	uint64_t frame_error[3] = {};

	for (;;)
	{
		if (!a.begin_frame() || !b.begin_frame())
			break;

		if (!a.read(Y[0].get(), num_luma_pixels))
			break;
		if (!b.read(Y[1].get(), num_luma_pixels))
			break;

		if (!a.read(Cb[0].get(), num_chroma_pixels))
			break;
		if (!b.read(Cb[1].get(), num_chroma_pixels))
			break;

		if (!a.read(Cr[0].get(), num_chroma_pixels))
			break;
		if (!b.read(Cr[1].get(), num_chroma_pixels))
			break;

		for (int i = 0; i < 3; i++)
			frame_error[i] = 0;

		for (int i = 0; i < num_luma_pixels; i++)
		{
			int dY = Y[0][i] - Y[1][i];
			dY *= dY;
			frame_error[0] += dY;
		}

		for (int i = 0; i < num_chroma_pixels; i++)
		{
			int dCb = Cb[0][i] - Cb[1][i];
			int dCr = Cr[0][i] - Cr[1][i];
			dCb *= dCb;
			dCr *= dCr;
			frame_error[1] += dCb;
			frame_error[2] += dCr;
		}

		frame_peak_signal[0] = num_luma_pixels * 255ull * 255ull;
		frame_peak_signal[1] = num_chroma_pixels * 255ull * 255ull;
		frame_peak_signal[2] = num_chroma_pixels * 255ull * 255ull;

		fprintf(stderr, "PSNR: (Y) %4.4f dB, (Cb) %4.4f dB, (Cr) %4.4f dB\n",
		        10.0 * std::log10(double(frame_peak_signal[0]) / double(frame_error[0])),
		        10.0 * std::log10(double(frame_peak_signal[1]) / double(frame_error[1])),
		        10.0 * std::log10(double(frame_peak_signal[2]) / double(frame_error[2])));

		for (int i = 0; i < 3; i++)
		{
			total_peak_signal[i] += frame_peak_signal[i];
			total_error[i] += frame_error[i];
		}
	}

	fprintf(stderr, "Overall PSNR: (Y) %4.4f dB, (Cb) %4.4f dB, (Cr) %4.4f dB\n",
	        10.0 * std::log10(double(total_peak_signal[0]) / double(total_error[0])),
	        10.0 * std::log10(double(total_peak_signal[1]) / double(total_error[1])),
	        10.0 * std::log10(double(total_peak_signal[2]) / double(total_error[2])));

}