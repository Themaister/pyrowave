struct VOut
{
	float2 uv : UV;
	float4 pos : SV_Position;
};

VOut VSMain(uint vid : SV_VertexID)
{
	VOut vout;
	if (vid == 0)
		vout.uv = float2(0, 0);
	else if (vid == 1)
		vout.uv = float2(0, 2);
	else
		vout.uv = float2(2, 0);

	vout.pos = float4(vout.uv * 2 - 1, 0, 1);
	vout.pos.y = -vout.pos.y;
	return vout;
}

struct PSOut
{
	float y : SV_Target0;
	float cb : SV_Target1;
	float cr : SV_Target2;
};

Texture2D<float3> RGB : register(t0);
SamplerState Samp : register(s0);

PSOut PSMain(VOut vout)
{
	PSOut psout;

	float3 rgb = RGB.Sample(Samp, vout.uv);
	static const float3x3 RGB2YUV = float3x3(0.2126, 0.7152, 0.0722, -0.114572, -0.385428, 0.5, 0.5, -0.454153, -0.0458471);
	float3 ycbcr = mul(RGB2YUV, rgb);

	ycbcr.yz += 128.0 / 255.0;

	psout.y = ycbcr.x;
	psout.cb = ycbcr.y;
	psout.cr = ycbcr.z;

	return psout;
}
