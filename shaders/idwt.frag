#version 450
// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

layout(location = 0) in vec2 vUV;

layout(location = 0) out mediump float oY;
#if OUTPUT_PLANES_MINUS_1 == 1
layout(location = 1) out mediump vec2 oCbCr;
#elif OUTPUT_PLANES_MINUS_1 == 2
layout(location = 1) out mediump float oCb;
layout(location = 2) out mediump float oCr;
#endif

layout(set = 0, binding = 0) uniform mediump texture2D uYEven;
layout(set = 0, binding = 1) uniform mediump texture2D uYOdd;
layout(set = 0, binding = 2) uniform mediump sampler uSampler;
#if INPUT_PLANES_MINUS_1 == 2
layout(set = 0, binding = 3) uniform mediump texture2D uCbEven;
layout(set = 0, binding = 4) uniform mediump texture2D uCrEven;
layout(set = 0, binding = 5) uniform mediump texture2D uCbOdd;
layout(set = 0, binding = 6) uniform mediump texture2D uCrOdd;
#elif INPUT_PLANES_MINUS_1 == 1
layout(set = 0, binding = 3) uniform mediump texture2D uCbCrEven;
layout(set = 0, binding = 4) uniform mediump texture2D uCbCrOdd;
#endif

#if (!INPUT_PLANES_MINUS_1 && OUTPUT_PLANES_MINUS_1) || (!OUTPUT_PLANES_MINUS_1 && INPUT_PLANES_MINUS_1)
// Degenerate scenario.
void main() {}
#else

// Direct and naive implementing of the CDF 9/7 synthesis filters.
// Optimized for the mobile GPUs which don't have any
// competent compute/shared memory performance whatsoever,
// i.e. anything not AMD/NV/Intel.

layout(constant_id = 0) const bool VERTICAL = false;
const ivec2 OFFSET_M2 = VERTICAL ? ivec2(0, -2) : ivec2(-2, 0);
const ivec2 OFFSET_M1 = VERTICAL ? ivec2(0, -1) : ivec2(-1, 0);
const ivec2 OFFSET_P1 = VERTICAL ? ivec2(0, +1) : ivec2(+1, 0);
const ivec2 OFFSET_P2 = VERTICAL ? ivec2(0, +2) : ivec2(+2, 0);

const float SYNTHESIS_LP_0 = 1.11508705;
const float SYNTHESIS_LP_1 = 0.591271763114;
const float SYNTHESIS_LP_2 = -0.057543526229;
const float SYNTHESIS_LP_3 = -0.091271763114;

const float SYNTHESIS_HP_0 = 0.602949018236;
const float SYNTHESIS_HP_1 = -0.266864118443;
const float SYNTHESIS_HP_2 = -0.078223266529;
const float SYNTHESIS_HP_3 = 0.016864118443;
const float SYNTHESIS_HP_4 = 0.026748757411;

void main()
{
	bool is_odd;

	if (VERTICAL)
		is_odd = (int(gl_FragCoord.y) & 1) != 0;
	else
		is_odd = (int(gl_FragCoord.x) & 1) != 0;

#define SAMPLE_COMPONENT(comp, swiz, T) \
	T comp##1 = T(textureLodOffset(sampler2D(u##comp##Odd, uSampler), vUV, 0.0, OFFSET_M2).swiz); \
	T comp##2 = T(textureLodOffset(sampler2D(u##comp##Even, uSampler), vUV, 0.0, OFFSET_M1).swiz); \
	T comp##3 = T(textureLodOffset(sampler2D(u##comp##Odd, uSampler), vUV, 0.0, OFFSET_M1).swiz); \
	T comp##4 = T(textureLod(sampler2D(u##comp##Even, uSampler), vUV, 0.0).swiz); \
	T comp##5 = T(textureLod(sampler2D(u##comp##Odd, uSampler), vUV, 0.0).swiz); \
	T comp##6 = T(textureLodOffset(sampler2D(u##comp##Even, uSampler), vUV, 0.0, OFFSET_P1).swiz); \
	T comp##7 = T(textureLodOffset(sampler2D(u##comp##Odd, uSampler), vUV, 0.0, OFFSET_P1).swiz); \
	T comp##8 = T(textureLodOffset(sampler2D(u##comp##Even, uSampler), vUV, 0.0, OFFSET_P2).swiz); \
	T comp##9 = T(textureLodOffset(sampler2D(u##comp##Odd, uSampler), vUV, 0.0, OFFSET_P2).swiz)

	SAMPLE_COMPONENT(Y, x, float);
#if INPUT_PLANES_MINUS_1 == 1
	SAMPLE_COMPONENT(CbCr, xy, vec2);
#elif INPUT_PLANES_MINUS_1 == 2
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

	// TODO: Deal with edge handling.

#if INPUT_PLANES_MINUS_1 > 0
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

#if OUTPUT_PLANES_MINUS_1 == 2
	oY = result.x;
	oCb = result.y;
	oCr = result.z;
#elif OUTPUT_PLANES_MINUS_1 == 1
	oY = result.x;
	oCbCr = result.yz;
#else
	oY = result;
#endif
}
#endif
