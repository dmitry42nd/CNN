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

  
  float* kernelData;
  float poolCoef = 0.5;
  int kernelWidth = 9; 
  int kernelHeight = 9;
  kernelData = new float[kernelWidth * kernelHeight];
  for (int i = 0; i < kernelWidth; ++i) {
    for (int j = 0; j < kernelHeight; ++j)
      kernelData[i + kernelWidth * j] = 0.f;
  }
  kernelData[4 + kernelWidth * 4] = 1.f;
  
  int kernelWidth2 = 7; 
  int kernelHeight2 = 7;
  float *kernelData2 = new float[kernelWidth2 * kernelHeight2];
  for (int i = 0; i < kernelWidth2; ++i) {
    for (int j = 0; j < kernelHeight2; ++j)
      if(i==0 || j==0 || i== kernelWidth2-1 || j== kernelHeight2-1)
        kernelData2[i + kernelWidth2 * j] = 0.8f;
      else if (i == 1 || j == 1 || i == kernelWidth2 - 2 || j == kernelHeight2 - 2)
        kernelData2[i + kernelWidth2 * j] = 0.4f;
      else
        kernelData2[i + kernelWidth2 * j] = 0.2f;
  }
  kernelData2[3 + kernelWidth2 * 3] = -5.f;

  int kernelWidth3 = 1;
  int kernelHeight3 = 1;
  float *kernelData3 = new float[kernelWidth3 * kernelHeight3];
  kernelData3[0] = 1.f;



  //common stuff >
  Mat inImage = imread("input.png");
  inImage.convertTo(inImage, CV_32SC3);

  if (inImage.empty()) {
    cout << "Image is empty" << endl;
    return 1;
  }
  int inImgWidth = inImage.size().width; 
  int inImgHeight = inImage.size().height;

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
  CNeuron cn1(kernelData2, kernelWidth2, context, device, commandQueue);
  CNeuron cn2(kernelData3, kernelWidth3, context, device, commandQueue);

  vector<shared_ptr<CNeuron>> cns0;
  int l1 = 8;
  for (int i = 0; i < l1; i ++)
    cns0.push_back(make_shared<CNeuron>(cn0));

  vector<shared_ptr<CNeuron>> cns1;
  int l2 = 4;
  for (int i = 0; i < l2; i++)
    cns1.push_back(make_shared<CNeuron>(cn1));

  vector<shared_ptr<CNeuron>> cns2;
  int l3 = 2;
  for (int i = 0; i < l3; i++)
    cns2.push_back(make_shared<CNeuron>(cn2));

//Pool Layer stuff
  PNeuron pn0(poolCoef, context, device, commandQueue);

//init layers
  shared_ptr<ILayer> iLayer(make_shared<ILayer>());
  shared_ptr<CLayer> cLayer0(make_shared<CLayer>(cns0));
  shared_ptr<PLayer> pLayer0(make_shared<PLayer>(make_shared<PNeuron>(pn0)));
  shared_ptr<CLayer> cLayer1(make_shared<CLayer>(cns1));
  shared_ptr<PLayer> pLayer1(make_shared<PLayer>(make_shared<PNeuron>(pn0)));
  shared_ptr<CLayer> cLayer2(make_shared<CLayer>(cns2));
//cnn run
  iLayer->activate(inImage, context);
  cLayer0->activate(iLayer->getFeatureMaps());
  
  pLayer0->activate(cLayer0->getFeatureMaps());
  cLayer1->activate(pLayer0->getFeatureMaps());
  pLayer1->activate(cLayer1->getFeatureMaps());
  cLayer2->activate(pLayer1->getFeatureMaps());
  
  char* x = new char[32];
  
  FeatureMaps out = cLayer2->getFeatureMaps();
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