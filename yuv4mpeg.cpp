// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT
#include "yuv4mpeg.hpp"
#include <string.h>

bool YUV4MPEGFile::open_read(const std::string &path)
{
	return open(path, Mode::Read);
}

bool YUV4MPEGFile::open_write(const std::string &path, const std::string &params_)
{
	params = params_;
	return open(path, Mode::Write);
}

bool YUV4MPEGFile::open(const std::string &path, Mode mode_)
{
	mode = mode_;
	file.reset(fopen(path.c_str(), mode == Mode::Read ? "rb" : "wb"));
	if (!file)
	{
		fprintf(stderr, "Failed to open %s\n", path.c_str());
		return false;
	}

	if (mode == Mode::Read)
	{
		char magic[11] = {};
		if (fread(magic, 1, sizeof(magic) - 1, file.get()) != sizeof(magic) - 1)
		{
			fprintf(stderr, "Failed to read magic.\n");
			return false;
		}

		if (strcmp(magic, "YUV4MPEG2 ") != 0)
		{
			fprintf(stderr, "Invalid magic\n");
			return false;
		}

		char c;
		while (fread(&c, 1, 1, file.get()) == 1 && c != '\n')
			params += c;
		params += '\n';
	}
	else
	{
		if (!write("YUV4MPEG2 ", 10))
			return false;

		if (!write(params.data(), params.size()))
			return false;
	}

	auto w_pos = params.find_first_of('W');
	if (w_pos == std::string::npos)
		return false;
	width = strtol(params.c_str() + w_pos + 1, nullptr, 0);

	auto h_pos = params.find_first_of('H');
	if (h_pos == std::string::npos)
		return false;
	height = strtol(params.c_str() + h_pos + 1, nullptr, 0);

	// Just assume normal 420 for now.
	return width > 0 && height > 0;
}

bool YUV4MPEGFile::begin_frame()
{
	if (mode == Mode::Write)
	{
		return fwrite("FRAME\n", 1, 6, file.get()) == 6;
	}
	else
	{
		std::string line;
		char c;

		while (fread(&c, 1, 1, file.get()) == 1 && c != '\n')
			line += c;

		return line == "FRAME";
	}
}

bool YUV4MPEGFile::read(void *pixels, size_t size)
{
	return fread(pixels, 1, size, file.get()) == size;
}

bool YUV4MPEGFile::write(const void *pixels, size_t size)
{
	return fwrite(pixels, 1, size, file.get()) == size;
}

int YUV4MPEGFile::get_width() const
{
	return width;
}

int YUV4MPEGFile::get_height() const
{
	return height;
}

const std::string &YUV4MPEGFile::get_params() const
{
	return params;
}