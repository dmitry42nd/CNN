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

void setKernels(int kernelSize, int nKernels, vector<float*>*kernelsData, string filePath) {
  ifstream fin_conv1;
  fin_conv1.exceptions(ifstream::failbit | ifstream::badbit);
  try {
    fin_conv1.open(filePath);
    for (int i = 0; i < nKernels; ++i)
    {
      float *weights_conv = new float[kernelSize];
      for (int j = 0; j < kernelSize; ++j)
      {
        string str;
        getline(fin_conv1, str, ',');
        weights_conv[j] = stod(str);
      }
      kernelsData->push_back(weights_conv);
    }
    fin_conv1.close();
  }
  catch (ifstream::failure error) {
    cerr << error.what() << endl;
  }
}


void prepareNeurons(int nNeurons, int nKernels, int kernelWidth, string filePath, vector<shared_ptr<CNeuron>> *cns) {
  printf("%s\n", filePath);
  int kernelSize = kernelWidth*kernelWidth;
  ifstream fin_conv;
  fin_conv.exceptions(ifstream::failbit | ifstream::badbit);

  try {
    fin_conv.open(filePath);
    for (int i = 0; i < nNeurons; ++i) {
      string line; 
      getline(fin_conv, line);
      stringstream iss(line);
      vector<float*> kernelsData;

      for (int j = 0; j < nKernels; ++j) {
        float *weights_conv = new float[kernelSize];
        for (int k = 0; k < kernelSize; ++k) {
          string str; getline(iss, str, ',');
          weights_conv[k] = stod(str);
        }
        kernelsData.push_back(weights_conv);
      }

      //create neuron based on kernel data
      CNeuron cn(kernelsData, kernelWidth, context, device, commandQueue);
      
      //create vector of neurons for conv layer1
      cns->push_back(make_shared<CNeuron>(cn));

      //should clean up that float* shit
    }
    fin_conv.close();
  }
  catch (ifstream::failure error) { cerr << error.what() << endl; }
}

int main(int argc, char** argv)
{
  //common stuff >
  Mat inImage = imread("inputM.jpg");
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

  //0 conv layer neurons
  int kernelWidth0 = 9;
  vector<shared_ptr<CNeuron>> cns0;
  prepareNeurons(64, 1, 9, "weights_conv1.csv", &cns0);

  //1 conv layer neurons
  int kernelWidth1 = 7; 
  vector<shared_ptr<CNeuron>> cns1;
  prepareNeurons(32, 64, 7, "weights_conv2.csv", &cns1);
  
  //2 conv layer neurons (22 layer in matlab code)
  int kernelWidth2 = 1;
  vector<shared_ptr<CNeuron>> cns2;
  prepareNeurons(16, 32, 1, "weights_conv22.csv", &cns2);

  //3 (Out) conv layer neurons
  int kernelWidth3 = 5;
  vector<shared_ptr<CNeuron>> cns3;
  prepareNeurons(1, 16, 5, "weights_conv3.csv", &cns3);

  //0 pool layer neurons
  float poolCoef0 = 1;
  float poolBias0 = 0;
  vector<shared_ptr<PNeuron>> pns0;
  for (int i = 0; i < 64; i++) {
    PNeuron pn0(poolBias0, context, device, commandQueue);
    pns0.push_back(make_shared<PNeuron>(pn0));
  }

  //1 pool layer neurons
  float poolCoef1 = 1;
  float poolBias1 = 0;
  vector<shared_ptr<PNeuron>> pns1;
  for (int i = 0; i < 32; i++) {
    PNeuron pn1(poolBias1, context, device, commandQueue);
    pns1.push_back(make_shared<PNeuron>(pn1));
  }

  //init layers
  shared_ptr<ILayer> iLayer(make_shared<ILayer>());
  shared_ptr<CLayer> cLayer0(make_shared<CLayer>(cns0));
  shared_ptr<PLayer> pLayer0(make_shared<PLayer>(pns0, poolCoef0));
  shared_ptr<CLayer> cLayer1(make_shared<CLayer>(cns1));
  shared_ptr<PLayer> pLayer1(make_shared<PLayer>(pns1, poolCoef1));
  shared_ptr<CLayer> cLayer2(make_shared<CLayer>(cns2));
  //without pool layer
  shared_ptr<CLayer> outLayer(make_shared<CLayer>(cns3));

  cout << "Layers are ready. Let's run!\n";
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

  cout << "done!\n";
  return 0;
}