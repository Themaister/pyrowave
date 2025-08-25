#version 450
// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

layout(location = 0) out vec2 vUV;

layout(push_constant) uniform Registers
{
	vec2 uv_offset;
};

void main()
{
	if (gl_VertexIndex == 0)
		vUV = vec2(0.0, 0.0);
	else if (gl_VertexIndex == 1)
		vUV = vec2(0.0, 2.0);
	else
		vUV = vec2(2.0, 0.0);

	gl_Position = vec4(vUV * 2.0 - 1.0, 0.0, 1.0);

	vUV += uv_offset;
}
