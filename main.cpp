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
  for (int i = 0; i < kernelWidth; ++i) {
    for (int j = 0; j < kernelHeight; ++j)
      kernelData[i + kernelWidth * j] = 0.1f;
  }
  
  float *kernelData2 = new float[kernelWidth * kernelHeight];
  for (int i = 0; i < kernelWidth; ++i) {
    for (int j = 0; j < kernelHeight; ++j)
      kernelData2[i + kernelWidth * j] = 0.1f;
  }

  //common stuff >
  Mat inImage = imread("input2.png");
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

//Input Layer
  ILayer iLayer(inImage);

//Convolution Layer
  CNeuron cn0(kernelData, kernelWidth, context, device, commandQueue);
  CNeuron cn1(kernelData2, kernelWidth, context, device, commandQueue);

  vector<shared_ptr<CNeuron>> cns;
  cns.push_back(make_shared<CNeuron>(cn0));
  cns.push_back(make_shared<CNeuron>(cn1));
  shared_ptr<CLayer> cLayer;// = make_shared<CLayer>(cns);
//Pool Layer
  PNeuron pn0(poolCoef, context, device, commandQueue);
  shared_ptr<PLayer> pLayer;// = make_shared<PLayer>(make_shared<PNeuron>(pn0));

  iLayer.activate(context);
  cLayer = make_shared<CLayer>(cns);
  cLayer->activate(iLayer.getFeatureMaps(), context);

  pLayer = make_shared<PLayer>(make_shared<PNeuron>(pn0));
  pLayer->activate(cLayer->getFeatureMaps(), context);

  cLayer = make_shared<CLayer>(cns);
  cLayer->activate(pLayer->getFeatureMaps(), context);
  
  cLayer->activate(cLayer->getFeatureMaps(), context);
  cLayer->activate(cLayer->getFeatureMaps(), context);
  cLayer->activate(cLayer->getFeatureMaps(), context);
  cLayer->activate(cLayer->getFeatureMaps(), context);
  
  char* x = new char[32];
  vector<mBuffer> out = cLayer->getFeatureMaps();
  for (int i = 0; i < out.size(); i++) {
    mBuffer o = out[i];
    Mat image = Mat::zeros(Size(o.width, o.height), CV_8UC3);
    commandQueue.enqueueReadBuffer(*o.data.get(), CL_TRUE, 0, sizeof(cl_uchar) * 3 * o.width * o.height, image.data);
    sprintf(x, "output%d.png", i);
    imwrite(x, image);
  }
  delete[] x;

  cout << "done!\n";
  return 0;
}