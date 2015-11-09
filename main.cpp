#define __CL_ENABLE_EXCEPTIONS

#include "layer.h"

#include <opencv\highgui.h>
#include <CL\cl.hpp>

#include <iostream>
#include <fstream>
#include <ctime>
#include <vector>

using namespace std;
using namespace cv;


int platformId = 0, deviceId = 0;

vector<cl::Platform> setPlatforms;
cl::Platform platform;
vector<cl::Device> setDevices;
cl::Device device;
cl::Context context;
cl::CommandQueue commandQueue;

/*
class CNN {
  private:
    vector<Layer*> layers;
  public:
    CNN(vector<Layer*> layers_) {
      layers = layers_;
    }

    void run() {
      printf("call ILayer\n");
      layers[0]->activate(NULL);
      for (int i = 1; i < layers.size(); i++) {
        layers[i]->activate(layers[i - 1]);
      }
    }
};
*/
int main(int argc, char** argv)
{
#if 0
  ILayer I("input.png");
  CLayer A(42); //vector<Neurons>
  SLayer B(2);
  OLayer O;
  vector<Layer*> layers;
  layers.push_back(&I);
  layers.push_back(&A);
  layers.push_back(&B);
  layers.push_back(&O);
  CNN cnn(layers);
  cnn.run();
#endif

  int kernelWidth, kernelHeight;
  float* kernelData;
  float poolCoef = 0.5;
  kernelWidth = 5; kernelHeight = 5;
  kernelData = new float[kernelWidth * kernelHeight];
  for (int i = 0; i < kernelWidth; ++i)
  {
    for (int j = 0; j < kernelHeight; ++j)
      kernelData[i + kernelWidth * j] = 0.1f;
  }
  
  //common stuff >
  Mat inImage = imread("input.png");
  Mat inImage2 = imread("input2.png");
  if (inImage.empty() || inImage2.empty()) {
    cout << "Image is empty" << endl;
    return 1;
  }
  int inImgWidth = inImage.size().width; 
  int inImgHeight = inImage.size().height;

  int inImgWidth2 = inImage2.size().width;
  int inImgHeight2 = inImage2.size().height;
    
  cl::Platform::get(&setPlatforms);
  platformId = 0; //or 1 for Dima K.
  platform = setPlatforms[platformId];

  platform.getDevices(CL_DEVICE_TYPE_ALL, &setDevices);
  device = setDevices[deviceId];
  context = cl::Context(device);
  
  commandQueue = cl::CommandQueue(context, device);

  cl::Buffer inImgBuf  = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_uchar) * 3 * inImgWidth * inImgHeight, (void*)inImage.data);
  cl::Buffer inImgBuf2 = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_uchar) * 3 * inImgWidth2 * inImgHeight2, (void*)inImage2.data);
  //< common stuff
  
  CNeuron cneuron(context, device, commandQueue);
  cneuron.setKernel(kernelData, kernelWidth);
  vector<CNeuron> cNeurons;

  cNeurons.push_back(cneuron);
  CLayer cLayer(make_shared<vector<CNeuron>>(cNeurons));
  (*cLayer.getNeurons().get())[0].convolve(&inImgBuf2, inImgWidth2, inImgHeight2);
  (*cLayer.getNeurons().get())[0].convolve(&inImgBuf, inImgWidth, inImgHeight);

  //cneuron.convolve(&inImgBuf2, inImgWidth2, inImgHeight2);
  //cneuron.convolve(&inImgBuf, inImgWidth, inImgHeight);

  const vector<Mat> &cmaps = (*cLayer.getNeurons().get())[0].getFeatureMaps();

  inImgWidth = cmaps[0].size().width;
  inImgHeight = cmaps[0].size().height;

  inImgWidth2 = cmaps[1].size().width;
  inImgHeight2 = cmaps[1].size().height;

  inImgBuf  = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_uchar) * 3 * inImgWidth * inImgHeight, (void*)cmaps[0].data);
  inImgBuf2 = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_uchar) * 3 * inImgWidth2 * inImgHeight2, (void*)cmaps[1].data);
  
  PNeuron pneuron(context, device, commandQueue);
  pneuron.setPoolCoef(poolCoef);
  pneuron.pool(&inImgBuf2, inImgWidth2, inImgHeight2);
  pneuron.pool(&inImgBuf, inImgWidth, inImgHeight);

  const vector<Mat> &maps = pneuron.getFeatureMaps();
  char* x = new char[32];
  for (int i = 0; i < maps.size(); i++) {
    sprintf(x, "output%d.png", i);
    imwrite(x, maps[i]);
  }
  delete[] x;
  
  cout << "done!\n";
  return 0;
}