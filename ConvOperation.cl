int sat(int min, int val, int max) {
  return val > min ? (val < max ? val : max) : min;
}

__kernel void Convolution(__global float* inImage,
                          int kernelWidth,
                          __global float* kernelData,
                          __global float* outImage) {
  int c = get_global_id(0),
      r = get_global_id(1);
  int cs = c - kernelWidth / 2,
      rs = r - kernelWidth / 2;

  int imgWidth  = get_global_size(0),
      imgHeight = get_global_size(1);

  for (int m = 0; m < kernelWidth; m++) {
    for (int n = 0; n < kernelWidth; n++) {
      for (int ch = 0; ch < 3; ch++) {
        outImage[3 * (r*imgWidth + c) + ch] += inImage[3 * (sat(0, rs + m, imgHeight - 1) * imgWidth + sat(0, cs + n, imgWidth)) + ch] *
                                               kernelData[n * kernelWidth + m];
      }
    }
  }
}