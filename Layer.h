#pragma once
#define __CL_ENABLE_EXCEPTIONS

#include <vector>
#include <opencv\highgui.h>
#include <CL\cl.hpp>

using namespace std;
using namespace cv;

class Neuron {
protected:
  vector<Mat> featureMaps;
public:
  const vector<Mat> &getFeatureMaps() const;
};

class ÑNeuron : public Neuron {
private:
  const string clConvFileName = "ConvOperation.cl";

  float     *kernelData;
  int        kernelWidth;
  int        kernelHeight;

  cl::Buffer kernelBuf;
  cl::Buffer convImgBuf;
  cl::Program::Sources source;
  cl::Program          program;
  cl::Kernel           kernel;
  string               kernelSrc;

  //static stuff
  const cl::Context      &context;
  const cl::Device       &device;
  const cl::CommandQueue &commandQueue;
  
  int init();
public:
  ÑNeuron(float *kernelData, int kernelWidth, const cl::Context &context, const cl::Device &device, const cl::CommandQueue &commandQueue);
  ÑNeuron(const cl::Context &context, const cl::Device &device, const cl::CommandQueue &commandQueue);

  int convolve(const cl::Buffer *inImgBuf, int inImgWidth, int inImgHeight);
  void setKernel(float *kernel, int width);
};

class PNeuron : public Neuron {
private:
  const string clPoolFileName = "MaxPooling.cl";
  //vector<Mat> featureMaps;

  float poolCoef;

  cl::Buffer poolImgBuf;
  cl::Program::Sources source;
  cl::Program          program;
  cl::Kernel           kernel;
  string               kernelSrc;

  //static stuff
  const cl::Context      &context;
  const cl::Device       &device;
  const cl::CommandQueue &commandQueue;

  int init();
public:
  PNeuron(const cl::Context &context, const cl::Device &device, const cl::CommandQueue &commandQueue);
  PNeuron(float poolCoef, const cl::Context &context, const cl::Device &device, const cl::CommandQueue &commandQueue);

  int pool(const cl::Buffer *inImgBuf, int inImgWidth, int inImgHeight);
  void setPoolCoef(float poolCoef);
};



class Layer {
protected:
  vector<Neuron> *neurons;
public:
  Layer();
  Layer(vector<Neuron> *neurons);
  virtual void activate(Layer *prevLayer_);
  vector<Neuron>* getNeurons();
};

class ILayer : public Layer {
public:
  ILayer(char* path_);
  void activate(Layer * prevLayer_) override;
};

class CLayer : public Layer {
//private:
public:
  CLayer(vector<Neuron> *neurons);
  void activate(Layer * prevLayer_) override;
};

class PLayer : public Layer {
public:
  PLayer(vector<Neuron> *neurons);
  void activate(Layer * prevLayer_) override;
};

class OLayer : public Layer {
public:
  OLayer();
  void activate(Layer * prevLayer_) override;
};