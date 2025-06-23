#version 450
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec3 FragColor;
layout(set = 0, binding = 0) uniform texture2D uY;
layout(set = 0, binding = 1) uniform texture2D uCb;
layout(set = 0, binding = 2) uniform texture2D uCr;
layout(set = 0, binding = 3) uniform sampler uSamp;

void main()
{
	float Y = textureLod(sampler2D(uY, uSamp), vUV, 0.0).x;
	float Cb = textureLod(sampler2D(uCb, uSamp), vUV, 0.0).x;
	float Cr = textureLod(sampler2D(uCr, uSamp), vUV, 0.0).x;
	Cb -= 0.5;
	Cr -= 0.5;

	mat3 bt709 = mat3(vec3(1.0, 1.0, 1.0),
			vec3(0.0, -0.11156702 / 0.6780, 1.8814),
			vec3(1.4746, -0.38737742 / 0.6780, 0.0));

	FragColor = bt709 * vec3(Y, Cb, Cr);
}
