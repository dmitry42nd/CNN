int sat(int min, int val, int max) {
  return val > min ? (val < max ? val : max) : min;
}

__kernel void Convolution(__global int* inImage,
                          int kernelWidth,
                          __global float* kernelData,
                          __global int* outImage,
                          int aggregate) {
  int c = get_global_id(0),
      r = get_global_id(1);
  int cs = c - kernelWidth / 2,
      rs = r - kernelWidth / 2;

  int imgWidth  = get_global_size(0),
      imgHeight = get_global_size(1);

  int pix[3] = {0, 0, 0};
  for (int m = 0; m < kernelWidth; m++) {
    for (int n = 0; n < kernelWidth; n++) {
      for (int ch = 0; ch < 3; ch++) {
        pix[ch] += inImage[3 * (sat(0, rs + m, imgHeight-1) * imgWidth + 
                                sat(0, cs + n, imgWidth)) + ch] * kernelData[m * kernelWidth + n];
      }
    }
  }

  outImage[3 * (r*imgWidth + c) + 0] += pix[0];
  outImage[3 * (r*imgWidth + c) + 1] += pix[1];
  outImage[3 * (r*imgWidth + c) + 2] += pix[2];

  if(aggregate) {

    float normDiv = 0;
    for (int i = 0; i < kernelWidth*kernelWidth; i++) {
      normDiv += kernelData[i];
    }
    float finalDiv = normDiv*aggregate;
    outImage[3 * (r*imgWidth + c) + 0] /= finalDiv;
    outImage[3 * (r*imgWidth + c) + 1] /= finalDiv;
    outImage[3 * (r*imgWidth + c) + 2] /= finalDiv;
  }
}