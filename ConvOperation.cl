__kernel void Convolution
(
	__global uchar* inImage,
	int kernelWidth,
	int kernelHeight,
	__global float* Kernel,
	__global uchar* outImage
)
{
	int outImgWidth = get_global_size(0), inImgWidth = outImgWidth + (kernelWidth - 1);
	int i = get_global_id(0), j = get_global_id(1),
		k = i + (kernelWidth - 1), l = j + (kernelHeight - 1);
	for (int m = 0; m < kernelWidth; ++m)
	{
		for (int n = 0; n < kernelHeight; ++n)
		{
			for (int channel = 0; channel < 3; ++channel)
				outImage[3 * (i + outImgWidth * j) + channel] += inImage[3 * ((k - m) + inImgWidth * (l - n)) + channel] * Kernel[m + kernelWidth * n];
		}
	}
}