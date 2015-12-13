__kernel void Pooling(__global float* inImage,
	                    float poolCoef,
                      float poolBias,
	                    __global float* outImage) {

	int outImgWidth = get_global_size(0), 
      coef        = convert_int(1 / poolCoef), 
      inImgWidth  = coef * outImgWidth,

		  i = get_global_id(0), 
      j = get_global_id(1),

		  m = coef * i, 
      n = coef * j,
		  a, b, c, d, fst_max, snd_max;

	for (int ch = 0; ch < 3; ++ch) {
		a = inImage[3 *  (m + inImgWidth * n) + ch];
		b = inImage[3 * ((m + 1) + inImgWidth * n) + ch];
		c = inImage[3 *  (m + inImgWidth * (n + 1)) + ch];
		d = inImage[3 * ((m + 1) + inImgWidth * (n + 1)) + ch];
		fst_max = ((a > b) ? a : b);
		snd_max = ((c > d) ? c : d);
    //float tmp = (((fst_max > snd_max) ? fst_max : snd_max) + poolBias);
    float tmp = a;// +poolBias;
		outImage[3 * (i + outImgWidth * j) + ch] = tmp > 0 ? tmp : 0;
	}
}