__kernel void Pooling
(
	__global int* inImage,
	float poolCoef,
	__global int* outImage
)
{
	int outImgWidth = get_global_size(0), koef = convert_int(1 / poolCoef), inImgWidth = koef * outImgWidth,
		i = get_global_id(0), j = get_global_id(1),
		m = koef * i, n = koef * j,
		a, b, c, d, fst_max, snd_max;
	for (int channel = 0; channel < 3; ++channel)
	{
		a = inImage[3 * (m + inImgWidth * n) + channel];
		b = inImage[3 * ((m + 1) + inImgWidth * n) + channel];
		c = inImage[3 * (m + inImgWidth * (n + 1)) + channel];
		d = inImage[3 * ((m + 1) + inImgWidth * (n + 1)) + channel];
		fst_max = ((a > b) ? a : b);
		snd_max = ((c > d) ? c : d);
		outImage[3 * (i + outImgWidth * j) + channel] = (fst_max > snd_max) ? fst_max : snd_max;
	}
}