// SPDX-License-Identifier: MIT
#pragma once

// Private benchmark experiment. Derive the variant from the exact shader used by
// the production backend so its bitstream math and missing-block zero fill cannot
// drift into a separate implementation. Fail if the canonical source changes.
#include <string>

namespace PyroWave
{
inline bool replace_unique_msl(std::string &source, const char *before, const char *after)
{
	const size_t position = source.find(before);
	if (position == std::string::npos || source.find(before, position + 1) != std::string::npos)
		return false;
	source.replace(position, std::char_traits<char>::length(before), after);
	return true;
}

inline bool build_batched_dequant_msl(const char *canonical_source, std::string &source)
{
	source = canonical_source;
	// Existing Z addressing would invalidate the assumption that Z is a free
	// band dimension, even if the declaration anchors still matched.
	if (source.find("gl_WorkGroupID.z") != std::string::npos)
		return false;
	return replace_unique_msl(source,
		"struct Registers\n{\n    int2 resolution;\n    int output_layer;\n    int block_offset_32x32;\n    int block_stride_32x32;\n};",
		"struct Registers\n{\n    int2 resolution;\n    int output_layer;\n    int block_offset_32x32;\n    int block_stride_32x32;\n};\n\n"
		"struct BatchedRegisters\n{\n    Registers bands[4];\n};") &&
	       replace_unique_msl(source,
		"kernel void pyrowave_wavelet_dequant(const device void* spvBufferAliasSet0Binding2 [[buffer(0)]], constant Registers& registers [[buffer(1)]],",
		"kernel void pyrowave_wavelet_dequant_batched(const device void* spvBufferAliasSet0Binding2 [[buffer(0)]], constant BatchedRegisters& batched_registers [[buffer(1)]],") &&
	       replace_unique_msl(source,
		"{\n    const device auto& payload_data_u8 = *(const device Payloads8*)spvBufferAliasSet0Binding2;",
		"{\n    constant Registers& registers = batched_registers.bands[gl_WorkGroupID.z];\n"
		"    const device auto& payload_data_u8 = *(const device Payloads8*)spvBufferAliasSet0Binding2;");
}
}
