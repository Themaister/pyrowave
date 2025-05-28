#version 450

layout(set = 0, binding = 0) uniform texture2D Y;
layout(set = 0, binding = 1) uniform texture2D Cb;
layout(set = 0, binding = 2) uniform texture2D Cr;
layout(set = 0, binding = 3) uniform sampler Samp;

layout(location = 0) out vec3 FragColor;
layout(location = 0) in vec2 vUV;

const mat3 yuv2rgb = mat3(
    vec3(1.0, 1.0, 1.0),
    vec3(0.0, -0.13397432 / 0.7152, 1.8556),
    vec3(1.5748, -0.33480248 / 0.7152, 0.0));

void main()
{
    float y = textureLod(sampler2D(Y, Samp), vUV, 0.0).x;
    float cb = textureLod(sampler2D(Cb, Samp), vUV, 0.0).x;
    float cr = textureLod(sampler2D(Cr, Samp), vUV, 0.0).x;
    FragColor = yuv2rgb * vec3(y, cb - 0.5, cr - 0.5);
}
