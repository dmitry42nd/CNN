#pragma once
#define __CL_ENABLE_EXCEPTIONS

#include <vector>
#include <memory>
#include <opencv\highgui.h>
#include <CL\cl.hpp>

using namespace std;
using namespace cv;

struct FeatureMaps {
  vector<shared_ptr<cl::Buffer>> buffers;
  int width;
  int height;
};

class Neuron {
protected:
  //const stuff
  const cl::Context      &context;
  const cl::Device       &device;
  const cl::CommandQueue &commandQueue;
public:
  Neuron(const cl::Context &context, const cl::Device &device, const cl::CommandQueue &commandQueue);
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

  shared_ptr<cl::Buffer> convolve(const FeatureMaps inFMaps);
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

  void pool(const FeatureMaps inFMaps, FeatureMaps *outFMaps);
  void setPoolCoef(float poolCoef);
};



class Layer {
protected:
  FeatureMaps featureMaps;
public:
  ~Layer();
  Layer();
  FeatureMaps getFeatureMaps() { return featureMaps; };
};

class ILayer : public Layer {
public:
  ILayer();
  void activate(Mat inImage, const cl::Context &context);
};

class OLayer : public Layer {
public:
  OLayer();
  void activate(FeatureMaps getFeatureMaps);
};

class HiddenLayer : public Layer {
protected:
public:
  virtual ~HiddenLayer();
  HiddenLayer();
  virtual void activate(FeatureMaps prevFeatureMaps);
};

class CLayer : public HiddenLayer {
protected:
  vector<shared_ptr<CNeuron>> neurons;
public:
  CLayer(vector<shared_ptr<CNeuron>> neurons);
  void activate(FeatureMaps prevFeatureMaps) override;
};

class PLayer : public HiddenLayer {
protected:
  shared_ptr<PNeuron> neuron;
public:
  PLayer(shared_ptr<PNeuron> neuron);
  void activate(FeatureMaps prevFeatureMaps) override;
};