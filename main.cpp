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
  Mat inImage = imread("input.png");
  inImage.convertTo(inImage, CV_32SC3);
  inImage.convertTo(inImage, CV_32SC3);
  Mat inImage2 = imread("input.png");
  inImage2.convertTo(inImage, CV_32SC3);

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
  //< common stuff

//Convolution Layer stuff
  CNeuron cn0(kernelData, kernelWidth, context, device, commandQueue);
  CNeuron cn1(kernelData2, kernelWidth, context, device, commandQueue);

  vector<shared_ptr<CNeuron>> cns;
  for (int i = 0; i < 50; i ++)
    cns.push_back(make_shared<CNeuron>(cn0));
  
  
//Pool Layer stuff
  PNeuron pn0(poolCoef, context, device, commandQueue);

//init layers
  shared_ptr<ILayer> iLayer(make_shared<ILayer>());
  shared_ptr<CLayer> cLayer(make_shared<CLayer>(cns));
  shared_ptr<PLayer> pLayer(make_shared<PLayer>(make_shared<PNeuron>(pn0)));

//cnn run
  iLayer->activate(inImage, context);
  cLayer->activate(iLayer->getFeatureMaps());
  //pLayer->activate(cLayer->getFeatureMaps());
  
  char* x = new char[32];
  FeatureMaps out = cLayer->getFeatureMaps();
  for (int i = 0; i < out.buffers.size(); i++) {
    cl::Buffer *o = out.buffers[i].get();
    Mat image = Mat::zeros(Size(out.width, out.height), CV_32SC3);
    commandQueue.enqueueReadBuffer(*o, CL_TRUE, 0, sizeof(cl_int) * 3 * out.width * out.height, image.data);
    sprintf(x, "output%d.png", i);
    image.convertTo(image, CV_8UC3);
    imwrite(x, image);
  }
  
  delete[] x;
  delete[] kernelData;
  delete[] kernelData2;

  cout << "done!\n";
  return 0;
}