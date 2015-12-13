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

  int            kernelWidth;
  vector<float*> kernelsData;

  cl::Buffer kernelBuf;

  cl::Program::Sources source;
  cl::Program          program;
  cl::Kernel           kernel;
  string               kernelSrc;

  int init();
public:
  CNeuron(vector<float*>kernelsData, int kernelWidth, const cl::Context &context, const cl::Device &device, const cl::CommandQueue &commandQueue);
  CNeuron(const cl::Context &context, const cl::Device &device, const cl::CommandQueue &commandQueue);

  shared_ptr<cl::Buffer> convolve(const FeatureMaps inFMaps);
  void setKernels(vector<float*>kernelsData, int width);
};

class PNeuron : public Neuron {
private:
  const string clPoolFileName = "MaxPooling.cl";

  float poolBias;

  cl::Program::Sources source;
  cl::Program          program;
  cl::Kernel           kernel;
  string               kernelSrc;

  int init();
public:
  PNeuron(const cl::Context &context, const cl::Device &device, const cl::CommandQueue &commandQueue);
  PNeuron(float poolBias, const cl::Context &context, const cl::Device &device, const cl::CommandQueue &commandQueue);

  shared_ptr<cl::Buffer> PNeuron::pool(const shared_ptr<cl::Buffer> buffer, int outWidth, int outHeight, float poolCoef);
  void setPoolCoef(float bias);
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
  vector<shared_ptr<PNeuron>> neurons;
  float poolCoef;
public:
  PLayer(vector<shared_ptr<PNeuron>> neurons, float poolCoef);
  void activate(FeatureMaps prevFeatureMaps) override;
};