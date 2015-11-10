#pragma once
#define __CL_ENABLE_EXCEPTIONS

#include <vector>
#include <memory>
#include <opencv\highgui.h>
#include <CL\cl.hpp>

using namespace std;
using namespace cv;

struct mBuffer {
  shared_ptr<cl::Buffer> data;
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

  mBuffer convolve(const mBuffer inImgBuf);
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

  mBuffer pool(const  mBuffer inImgBuf);
  void setPoolCoef(float poolCoef);
};



class Layer {
protected:
  vector<mBuffer> featureMaps;
public:
  virtual ~Layer();
  Layer();
  virtual void activate(vector<mBuffer> prevFeatureMaps, const cl::Context &context);
  vector<mBuffer> getFetureMaps () { return featureMaps; };
};

class ILayer : public Layer {
protected:
  Mat inImage;
public:
  ILayer(Mat inImage);
  void activate(const cl::Context &context);
};

class OLayer : public Layer {
public:
  OLayer();
  void activate(vector<mBuffer> prevFeatureMaps, const cl::Context &context) override;
};

class CLayer : public Layer {
protected:
  vector<shared_ptr<CNeuron>> neurons;
public:
  CLayer(vector<shared_ptr<CNeuron>> neurons);
  void activate(vector<mBuffer> prevFeatureMaps, const cl::Context &context) override;
};

class PLayer : public Layer {
protected:
  PNeuron neurons;
public:
  PLayer(PNeuron neurons);
  void activate(vector<mBuffer> prevFeatureMaps, const cl::Context &context) override;
};