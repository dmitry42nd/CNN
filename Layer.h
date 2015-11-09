#pragma once
#define __CL_ENABLE_EXCEPTIONS

#include <vector>
#include <memory>
#include <opencv\highgui.h>
#include <CL\cl.hpp>

using namespace std;
using namespace cv;

class Neuron {
protected:
  vector<Mat> featureMaps;
public:
  const vector<Mat> &getFeatureMaps();
};

class CNeuron : public Neuron {
private:
  const string clConvFileName = "ConvOperation.cl";

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
  CNeuron(float *kernelData, int kernelWidth, const cl::Context &context, const cl::Device &device, const cl::CommandQueue &commandQueue);
  CNeuron(const cl::Context &context, const cl::Device &device, const cl::CommandQueue &commandQueue);

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



template<typename NeuronType>
class Layer {
protected:
    shared_ptr<vector<NeuronType>> neurons;
public:
  Layer();
  virtual void activate(Layer *prevLayer_);
  shared_ptr<vector<NeuronType>> getNeurons();
  void setNeurons(shared_ptr<vector<NeuronType>> neurons);
};

class ILayer : public Layer<Neuron> {
public:
  ILayer(char* path_);
  void activate(Layer * prevLayer_) override;
  shared_ptr<vector<Neuron>> getNeurons();
};

class OLayer : public Layer<Neuron> {
public:
  OLayer();
  void activate(Layer * prevLayer_) override;
  shared_ptr<vector<Neuron>> getNeurons();
};


class CLayer : public Layer<CNeuron> {
public:
  CLayer(shared_ptr<vector<CNeuron>> neurons);
  void activate(Layer * prevLayer_) override;
  shared_ptr<vector<CNeuron>> getNeurons();
};

class PLayer : public Layer<PNeuron> {
public:
  PLayer(shared_ptr<vector<PNeuron>> neurons);
  void activate(Layer * prevLayer_) override;
  shared_ptr<vector<PNeuron>> getNeurons();
};