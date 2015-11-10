int sat(int min, int val, int max) {
  return val > min ? (val < max ? val : max) : min;
}

__kernel void Convolution(__global uchar* inImage,
  int kernelWidth,
  int kernelHeight,
  __global float* kernelData,
  __global uchar* outImage) {
  int r = get_global_id(0),
      c = get_global_id(1);

  int  imgWidth = get_global_size(0),
      imgHeight = get_global_size(1);

  int kernelMid = kernelHeight*kernelWidth / 2;

  float div = 0;
  for (int i = 0; i < kernelHeight*kernelWidth; i++) {
    div += kernelData[i];
  }
  int pix[3] = { 0,0,0 };
  for (int m = -kernelHeight / 2; m <= kernelHeight / 2; m++) {
    for (int n = -kernelWidth / 2; n <= kernelWidth / 2; n++) {
      for (int ch = 0; ch < 3; ch++) {
        pix[ch] += inImage[3 * (sat(0, r + m, imgHeight) * imgWidth + sat(0, c + n, imgWidth)) + ch] * kernelData[kernelMid + m * kernelWidth + n];
      }
    }
  }
  outImage[3 * (r*imgWidth + c)] = pix[0] / div;
  outImage[3 * (r*imgWidth + c) + 1] = pix[1] / div;
  outImage[3 * (r*imgWidth + c) + 2] = pix[2] / div;
}