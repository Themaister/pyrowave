// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT
#pragma once

#include <memory>
#include <string>
#include <stdio.h>
#include <stddef.h>

class YUV4MPEGFile
{
public:
	bool open_read(const std::string &path);
	bool open_write(const std::string &path, const std::string &params);
	const std::string &get_params() const;

	int get_width() const;
	int get_height() const;

	bool begin_frame();
	bool write(const void *pixels, size_t size);
	bool read(void *pixels, size_t size);

	enum class Format
	{
		YUV420P,
		YUV420P16
	};

	Format get_format() const;
	bool is_full_range() const;

private:
	enum class Mode { Read, Write };
	bool open(const std::string &path, Mode mode);
	struct FileDeleter { void operator()(FILE *f) { if (f) fclose(f); } };
	std::unique_ptr<FILE, FileDeleter> file;
	int width = 0, height = 0;
	std::string params;
	Mode mode = {};
	Format format = {};
	bool full_range = false;
	float unorm_scale = 1.0f;
};