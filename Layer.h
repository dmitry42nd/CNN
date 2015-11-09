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

  //const stuff
  const cl::Context      &context;
  const cl::Device       &device;
  const cl::CommandQueue &commandQueue;
public:
  Neuron(const cl::Context &context, const cl::Device &device, const cl::CommandQueue &commandQueue);
  const vector<Mat> &getFeatureMaps();
  void setFeatureMap(vector<Mat> featureMaps);
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

  int init();
public:
  CNeuron(float *kernelData, int kernelWidth, const cl::Context &context, const cl::Device &device, const cl::CommandQueue &commandQueue);
  CNeuron(const cl::Context &context, const cl::Device &device, const cl::CommandQueue &commandQueue);

  int convolve(const shared_ptr<cl::Buffer> inImgBuf, int inImgWidth, int inImgHeight);
  void setKernel(float *kernel, int width);
};

class PNeuron : public Neuron {
private:
  const string clPoolFileName = "MaxPooling.cl";

  float poolCoef;

  cl::Buffer poolImgBuf;
  cl::Program::Sources source;
  cl::Program          program;
  cl::Kernel           kernel;
  string               kernelSrc;

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
  vector<shared_ptr<NeuronType>> neurons;
public:
  virtual ~Layer();
  Layer();
  virtual void activate(vector<shared_ptr<Neuron>> prevNeurons, const cl::Context &context);
  virtual vector<shared_ptr<NeuronType>> getNeurons(); //why it must be virtual?
  void setNeurons(vector<shared_ptr<NeuronType>> neurons);
};

class ILayer : public Layer<Neuron> {
public:
  ILayer(char* path_);
  void activate(vector<shared_ptr<Neuron>> prevNeurons, const cl::Context &context) override;
  //vector<shared_ptr<Neuron>> getNeurons();
};

class OLayer : public Layer<Neuron> {
public:
  OLayer();
  void activate(vector<shared_ptr<Neuron>> prevNeurons, const cl::Context &context) override;
  //vector<shared_ptr<Neuron>> getNeurons();
};

class CLayer : public Layer<CNeuron> {
public:
  CLayer(vector<shared_ptr<CNeuron>> neurons);
  void activate(vector<shared_ptr<Neuron>> prevNeurons, const cl::Context &context) override;
  //vector<shared_ptr<CNeuron>> getNeurons();
};

class PLayer : public Layer<PNeuron> {
public:
  PLayer(vector<shared_ptr<PNeuron>> neurons);
  void activate(vector<shared_ptr<Neuron>> prevNeurons, const cl::Context &context) override;
  //vector<shared_ptr<PNeuron>> getNeurons();
};