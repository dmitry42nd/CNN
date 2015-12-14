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

void prepareCNeurons(int nNeurons, int nKernels, int kernelWidth, string filePath, vector<shared_ptr<CNeuron>> *cns) {
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
          string str;
          getline(iss, str, ',');
          weights_conv[k] = stof(str);
        }
        kernelsData.push_back(weights_conv);
      }

      //create neuron based on kernel data
      CNeuron cn(kernelsData, kernelWidth, context, device, commandQueue);
      
      //create vector of neurons for convolution layer
      cns->push_back(make_shared<CNeuron>(cn));

      //should clean up that float* shit
    }
    fin_conv.close();
  }
  catch (ifstream::failure error) { cerr << error.what() << endl; }
}

void preparePNeurons(int nNeurons, string filePath, vector<shared_ptr<PNeuron>> *pns) {
  ifstream fin_pool;
  fin_pool.exceptions(ifstream::failbit | ifstream::badbit);

  try {
    fin_pool.open(filePath);
    for (int i = 0; i < nNeurons; ++i) {
      string strBias;
      getline(fin_pool, strBias);
      float bias = stof(strBias)*255.;

      //create neuron based on bias
      PNeuron pn(bias, context, device, commandQueue);

      //create vector of neurons for pool layer
      pns->push_back(make_shared<PNeuron>(pn));
    }
    fin_pool.close();
  }
  catch (ifstream::failure error) { cerr << error.what() << endl; }
}


int main(int argc, char** argv)
{
  //common stuff >
  Mat inImage = imread("data/input.jpg");
  if (inImage.empty()) {
    cout << "Image is empty" << endl;
    return 1;
  }
  
  inImage.convertTo(inImage, CV_32FC3);
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

  cout << "CNN layers are preparing...\n";
  //0 conv layer neurons
  int kernelWidth0 = 9;
  vector<shared_ptr<CNeuron>> cns0;
  prepareCNeurons(64, 1, 9, "data/weights_conv1.csv", &cns0);

  //1 conv layer neurons
  int kernelWidth1 = 7; 
  vector<shared_ptr<CNeuron>> cns1;
  prepareCNeurons(32, 64, 7, "data/weights_conv2.csv", &cns1);
  
  //2 conv layer neurons (22 layer in matlab code)
  int kernelWidth2 = 1;
  vector<shared_ptr<CNeuron>> cns2;
  prepareCNeurons(16, 32, 1, "data/weights_conv22.csv", &cns2);

  //3 (Out) conv layer neurons
  int kernelWidth3 = 5;
  vector<shared_ptr<CNeuron>> cns3;
  prepareCNeurons(1, 16, 5, "data/weights_conv3.csv", &cns3);

  //0 pool layer neurons
  vector<shared_ptr<PNeuron>> pns0;
  preparePNeurons(64, "data/biases_conv1.csv", &pns0);
  
  //1 pool layer neurons
  vector<shared_ptr<PNeuron>> pns1;
  preparePNeurons(32, "data/biases_conv2.csv", &pns1);

  //2 pool layer neurons
  vector<shared_ptr<PNeuron>> pns2;
  preparePNeurons(16, "data/biases_conv22.csv", &pns2);

  //3 pool layer neurons
  vector<shared_ptr<PNeuron>> pns3;
  preparePNeurons(1, "data/biases_conv3.csv", &pns3);


  //init layers
  shared_ptr<ILayer> iLayer(make_shared<ILayer>());
  shared_ptr<CLayer> cLayer0(make_shared<CLayer>(cns0));
  shared_ptr<PLayer> pLayer0(make_shared<PLayer>(pns0, 1.f));
  shared_ptr<CLayer> cLayer1(make_shared<CLayer>(cns1));
  shared_ptr<PLayer> pLayer1(make_shared<PLayer>(pns1, 1.f));
  shared_ptr<CLayer> cLayer2(make_shared<CLayer>(cns2));
  shared_ptr<PLayer> pLayer2(make_shared<PLayer>(pns2, 1.f));
  shared_ptr<CLayer> outCLayer(make_shared<CLayer>(cns3));
  shared_ptr<PLayer> outPLayer(make_shared<PLayer>(pns3, 1.f));

  cout << "Layers are ready. Let's run!\n";
  //cnn run
  iLayer->activate(inImage, context);
  cLayer0->activate(iLayer->getFeatureMaps());
  pLayer0->activate(cLayer0->getFeatureMaps());
  
  cLayer1->activate(pLayer0->getFeatureMaps());
  pLayer1->activate(cLayer1->getFeatureMaps());
  cLayer2->activate(pLayer1->getFeatureMaps());
  pLayer2->activate(cLayer2->getFeatureMaps());
  outCLayer->activate(pLayer2->getFeatureMaps());
  outPLayer->activate(outCLayer->getFeatureMaps());

  char* x = new char[32];
  
  FeatureMaps out = outPLayer->getFeatureMaps();
  for (size_t i = 0; i < out.buffers.size(); i++) {
    cl::Buffer *o = out.buffers[i].get();
    Mat image = Mat::zeros(Size(out.width, out.height), CV_32FC3);
    commandQueue.enqueueReadBuffer(*o, CL_TRUE, 0, sizeof(cl_float) * 3 * out.width * out.height, image.data);
    sprintf(x, "output%d.png", i);
    image.convertTo(image, CV_8UC3);
    imwrite(x, image);
  }
  
  delete[] x;

  cout << "Done!\n";
  return 0;
}