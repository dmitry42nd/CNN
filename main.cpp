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

int main(int argc, char** argv)
{
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

  //1st layer neuron kernel example
  int kernelWidth0 = 9; 
  int kernelHeight0 = 9;
  float* kernelData0 = new float[kernelWidth0 * kernelHeight0];

  for (int i = 0; i < kernelWidth0; ++i) {
    for (int j = 0; j < kernelHeight0; ++j)
      kernelData0[i + kernelWidth0 * j] = 0.f;
  }
  kernelData0[4 + kernelWidth0 * 4] = 1.f;
  
  //2nd layer neuron kernel example
  int kernelWidth1 = 7; 
  int kernelHeight1 = 7;
  float *kernelData1 = new float[kernelWidth1 * kernelHeight1];

  for (int i = 0; i < kernelWidth1; ++i) {
    for (int j = 0; j < kernelHeight1; ++j)
      if(i==0 || j==0 || i== kernelWidth1-1 || j== kernelHeight1-1)
        kernelData1[i + kernelWidth1 * j] = 0.8f;
      else if (i == 1 || j == 1 || i == kernelWidth1 - 2 || j == kernelHeight1 - 2)
        kernelData1[i + kernelWidth1 * j] = 0.4f;
      else
        kernelData1[i + kernelWidth1 * j] = 0.2f;
  }
  kernelData1[3 + kernelWidth1 * 3] = -5.f;

  //3d layer neuron kernel example (22 layer in matlab code)
  int kernelWidth2 = 1;
  int kernelHeight2 = 1;
  float *kernelData2 = new float[kernelWidth2 * kernelHeight2];
  kernelData2[0] = 1.f;

  //Out layer neuron example
  int kernelWidth3 = 5;
  int kernelHeight3 = 5;
  float *kernelData3 = new float[kernelWidth3 * kernelHeight3];

  for (int i = 0; i < kernelWidth3; ++i) {
    for (int j = 0; j < kernelHeight3; ++j)
      kernelData3[i + kernelWidth3 * j] = 0.2f;
  }

  //common pool coefficient
  float poolCoef = 0.5;

//Convolution Layers stuff
  //create neuron based on kernel data
  CNeuron cn0(kernelData0, kernelWidth0, context, device, commandQueue);

  //create vector of neurons for conv layer1
  vector<shared_ptr<CNeuron>> cns0;
  int l1 = 8; //64
  for (int i = 0; i < l1; i ++)
    cns0.push_back(make_shared<CNeuron>(cn0));

  CNeuron cn1(kernelData1, kernelWidth1, context, device, commandQueue);
  //create vector of neurons for conv layer2
  vector<shared_ptr<CNeuron>> cns1;
  int l2 = 4; //32
  for (int i = 0; i < l2; i++)
    cns1.push_back(make_shared<CNeuron>(cn1));

  //create vector of neurons for conv layer3
  CNeuron cn2(kernelData2, kernelWidth2, context, device, commandQueue);
  vector<shared_ptr<CNeuron>> cns2;
  int l3 = 2; //16
  for (int i = 0; i < l3; i++)
    cns2.push_back(make_shared<CNeuron>(cn2));

  //create vector of neurons for out layer
  CNeuron cn3(kernelData3, kernelWidth3, context, device, commandQueue);
  vector<shared_ptr<CNeuron>> cns3;
  cns3.push_back(make_shared<CNeuron>(cn3));

  //Pool Layer stuff
  PNeuron pn0(poolCoef, context, device, commandQueue);

  //init layers
  shared_ptr<ILayer> iLayer(make_shared<ILayer>());

  shared_ptr<CLayer> cLayer0(make_shared<CLayer>(cns0));
  shared_ptr<PLayer> pLayer0(make_shared<PLayer>(make_shared<PNeuron>(pn0)));

  shared_ptr<CLayer> cLayer1(make_shared<CLayer>(cns1));
  shared_ptr<PLayer> pLayer1(make_shared<PLayer>(make_shared<PNeuron>(pn0)));

  shared_ptr<CLayer> cLayer2(make_shared<CLayer>(cns2));
  //without pool layer

  shared_ptr<CLayer> outLayer(make_shared<CLayer>(cns3));

  //cnn run
  iLayer->activate(inImage, context);
  cLayer0->activate(iLayer->getFeatureMaps());
  pLayer0->activate(cLayer0->getFeatureMaps());
  cLayer1->activate(pLayer0->getFeatureMaps());
  pLayer1->activate(cLayer1->getFeatureMaps());
  cLayer2->activate(pLayer1->getFeatureMaps());
  outLayer->activate(cLayer2->getFeatureMaps());
  
  char* x = new char[32];
  
  FeatureMaps out = outLayer->getFeatureMaps();
  for (int i = 0; i < out.buffers.size(); i++) {
    cl::Buffer *o = out.buffers[i].get();
    Mat image = Mat::zeros(Size(out.width, out.height), CV_32SC3);
    commandQueue.enqueueReadBuffer(*o, CL_TRUE, 0, sizeof(cl_int) * 3 * out.width * out.height, image.data);
    sprintf(x, "output%d.png", i);
    image.convertTo(image, CV_8UC3);
    imwrite(x, image);
  }
  
  delete[] x;
  delete[] kernelData0;
  delete[] kernelData1;

  cout << "done!\n";
  return 0;
}