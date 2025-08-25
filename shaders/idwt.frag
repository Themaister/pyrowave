#version 450
// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

layout(location = 0) in vec2 vUV;

#if CHROMA_CONFIG == 0
#define OUTPUT_PLANES 1
#define INPUT_PLANES 1
#elif CHROMA_CONFIG == 1
#define OUTPUT_PLANES 2
#define INPUT_PLANES 3
#elif CHROMA_CONFIG == 2
#define OUTPUT_PLANES 3
#define INPUT_PLANES 2
#else
#error "Invalid chroma config"
#endif

layout(location = 0) out mediump float oY;
#if OUTPUT_PLANES == 2
layout(location = 1) out mediump vec2 oCbCr;
#elif OUTPUT_PLANES == 3
layout(location = 1) out mediump float oCb;
layout(location = 2) out mediump float oCr;
#endif

layout(set = 0, binding = 0) uniform mediump texture2D uYEven;
layout(set = 0, binding = 1) uniform mediump texture2D uYOdd;
layout(set = 0, binding = 2) uniform mediump sampler uSampler;
#if INPUT_PLANES == 3
layout(set = 0, binding = 3) uniform mediump texture2D uCbEven;
layout(set = 0, binding = 4) uniform mediump texture2D uCbOdd;
layout(set = 0, binding = 5) uniform mediump texture2D uCrEven;
layout(set = 0, binding = 6) uniform mediump texture2D uCrOdd;
#elif INPUT_PLANES == 2
layout(set = 0, binding = 3) uniform mediump texture2D uCbCrEven;
layout(set = 0, binding = 4) uniform mediump texture2D uCbCrOdd;
#endif

// Direct and naive implementing of the CDF 9/7 synthesis filters.
// Optimized for the mobile GPUs which don't have any
// competent compute/shared memory performance whatsoever,
// i.e. anything not AMD/NV/Intel.

layout(constant_id = 0) const bool VERTICAL = false;
layout(constant_id = 1) const bool FINAL_Y = false;
layout(constant_id = 2) const bool FINAL_CBCR = false;
layout(constant_id = 3) const int EDGE_CONDITION = 0;
const ivec2 OFFSET_M2 = VERTICAL ? ivec2(0, 0) : ivec2(0, 0);
const ivec2 OFFSET_M1 = VERTICAL ? ivec2(0, 1) : ivec2(1, 0);
const ivec2 OFFSET_C  = VERTICAL ? ivec2(0, 2) : ivec2(2, 0);
const ivec2 OFFSET_P1 = VERTICAL ? ivec2(0, 3) : ivec2(3, 0);
const ivec2 OFFSET_P2 = VERTICAL ? ivec2(0, 4) : ivec2(4, 0);

const float SYNTHESIS_LP_0 = 1.11508705;
const float SYNTHESIS_LP_1 = 0.591271763114;
const float SYNTHESIS_LP_2 = -0.057543526229;
const float SYNTHESIS_LP_3 = -0.091271763114;

const float SYNTHESIS_HP_0 = 0.602949018236;
const float SYNTHESIS_HP_1 = -0.266864118443;
const float SYNTHESIS_HP_2 = -0.078223266529;
const float SYNTHESIS_HP_3 = 0.016864118443;
const float SYNTHESIS_HP_4 = 0.026748757411;

layout(push_constant) uniform Registers
{
	vec2 uv_offset;
	int aligned_transform_size;
};

void main()
{
	int integer_coord;
	if (VERTICAL)
		integer_coord = int(gl_FragCoord.y);
	else
		integer_coord = int(gl_FragCoord.x);

	bool is_odd = (integer_coord & 1) != 0;

#define SAMPLE_COMPONENT(comp, swiz, T) \
	T comp##1 = T(textureLodOffset(sampler2D(u##comp##Odd, uSampler), vUV, 0.0, OFFSET_M2).swiz); \
	T comp##2 = T(textureLodOffset(sampler2D(u##comp##Even, uSampler), vUV, 0.0, OFFSET_M1).swiz); \
	T comp##3 = T(textureLodOffset(sampler2D(u##comp##Odd, uSampler), vUV, 0.0, OFFSET_M1).swiz); \
	T comp##4 = T(textureLodOffset(sampler2D(u##comp##Even, uSampler), vUV, 0.0, OFFSET_C).swiz); \
	T comp##5 = T(textureLodOffset(sampler2D(u##comp##Odd, uSampler), vUV, 0.0, OFFSET_C).swiz); \
	T comp##6 = T(textureLodOffset(sampler2D(u##comp##Even, uSampler), vUV, 0.0, OFFSET_P1).swiz); \
	T comp##7 = T(textureLodOffset(sampler2D(u##comp##Odd, uSampler), vUV, 0.0, OFFSET_P1).swiz); \
	T comp##8 = T(textureLodOffset(sampler2D(u##comp##Even, uSampler), vUV, 0.0, OFFSET_P2).swiz); \
	T comp##9 = T(textureLodOffset(sampler2D(u##comp##Odd, uSampler), vUV, 0.0, OFFSET_P2).swiz)

	SAMPLE_COMPONENT(Y, x, float);
#if INPUT_PLANES == 2
	SAMPLE_COMPONENT(CbCr, xy, vec2);
#elif INPUT_PLANES == 3
	SAMPLE_COMPONENT(Cb, x, float);
	SAMPLE_COMPONENT(Cr, x, float);
	vec2 CbCr1 = vec2(Cb1, Cr1);
	vec2 CbCr2 = vec2(Cb2, Cr2);
	vec2 CbCr3 = vec2(Cb3, Cr3);
	vec2 CbCr4 = vec2(Cb4, Cr4);
	vec2 CbCr5 = vec2(Cb5, Cr5);
	vec2 CbCr6 = vec2(Cb6, Cr6);
	vec2 CbCr7 = vec2(Cb7, Cr7);
	vec2 CbCr8 = vec2(Cb8, Cr8);
	vec2 CbCr9 = vec2(Cb9, Cr9);
#endif

	if (EDGE_CONDITION < 0)
	{
		// The mirroring rules are particular.
		// For odd inputs we can rely on the mirrored sampling to get intended behavior.
		if (integer_coord == 0)
		{
			// Y4 is the pivot.
			Y2 = Y6;
#if INPUT_SAMPLES > 1
			CbCr2 = CbCr6;
#endif
		}
	}
	else if (EDGE_CONDITION > 0)
	{
		if (integer_coord + 2 >= aligned_transform_size)
		{
			// We're on the last two pixels.
			// Y5 is the pivot. LP inputs behave as expected when using mirroring.
			Y7 = Y3;
			Y9 = Y1;
#if INPUT_SAMPLES > 1
			CbCr7 = CbCr3;
			CbCr9 = CbCr1;
#endif
		}
		else if (integer_coord + 4 >= aligned_transform_size)
		{
			// Y7 is the pivot.
			Y9 = Y5;
#if INPUT_SAMPLES > 1
			CbCr9 = CbCr5;
#endif
		}
	}

#if INPUT_PLANES > 1
#define AccumT vec3
#define GenInput(comp) vec3(Y##comp, CbCr##comp)
#else
#define AccumT float
#define GenInput(comp) Y##comp
#endif

	AccumT C0, C1, C2, C3, C4;
	float W0, W1, W2, W3, W4;

	// Not ideal, but gotta do what we gotta do.
	// GPU will have to take both paths here,
	// but at least we avoid dynamic load-store which is RIP perf on these chips ...
	if (is_odd)
	{
		C0 = GenInput(5);
		C1 = GenInput(4) + GenInput(6);
		C2 = GenInput(3) + GenInput(7);
		C3 = GenInput(2) + GenInput(8);
		C4 = GenInput(1) + GenInput(9);

		W0 = SYNTHESIS_HP_0;
		W1 = SYNTHESIS_LP_1;
		W2 = SYNTHESIS_HP_2;
		W3 = SYNTHESIS_LP_3;
		W4 = SYNTHESIS_HP_4;
	}
	else
	{
		C0 = GenInput(4);
		C1 = GenInput(3) + GenInput(5);
		C2 = GenInput(2) + GenInput(6);
		C3 = GenInput(1) + GenInput(7);
		C4 = AccumT(0.0);

		W0 = SYNTHESIS_LP_0;
		W1 = SYNTHESIS_HP_1;
		W2 = SYNTHESIS_LP_2;
		W3 = SYNTHESIS_HP_3;
		W4 = 0.0;
	}

	AccumT result = C0 * W0 + C1 * W1 + C2 * W2 + C3 * W3 + C4 * W4;

#if OUTPUT_PLANES == 3
	oY = result.x;
	oCb = result.y;
	oCr = result.z;
#elif OUTPUT_PLANES == 2
	oY = result.x;
	oCbCr = result.yz;
#else
	oY = result;
#endif

	if (FINAL_Y)
		oY += 0.5;

	if (FINAL_CBCR)
	{
#if OUTPUT_PLANES == 3
		oCb += 0.5;
		oCr += 0.5;
#elif OUTPUT_PLANES == 2
		oCbCr += 0.5;
#endif
	}
}
