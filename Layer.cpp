#include "Layer.h"

#include <fstream>
#include <iostream>

Neuron::Neuron(const cl::Context & context, const cl::Device & device, const cl::CommandQueue & commandQueue) :
  context(context),
  device(device),
  commandQueue(commandQueue) {
}


CNeuron::CNeuron(float *kernelData, int kernelWidth, const cl::Context &context, const cl::Device &device, const cl::CommandQueue &commandQueue) : 
  Neuron(context, device, commandQueue),
  kernelWidth(kernelWidth),
  kernelHeight(kernelWidth) {
  setKernel(kernelData, kernelWidth);
  init();
}

CNeuron::CNeuron(const cl::Context &context, const cl::Device &device, const cl::CommandQueue &commandQueue) :
  Neuron(context, device, commandQueue) {
  init();
}

int CNeuron::init() {
  ifstream fin;
  fin.exceptions(ifstream::failbit | ifstream::badbit);
  try {
    fin.open(clConvFileName);
    kernelSrc = string(istreambuf_iterator<char>(fin), (istreambuf_iterator<char>()));
    source = cl::Program::Sources(1, std::make_pair(kernelSrc.c_str(), kernelSrc.length() + 1));
    fin.close();
  }
  catch (ifstream::failure error) {
    cerr << "Exception opening/reading/closing 'ConvOperation.cl'" << endl;
    return 1;
  }

  program = cl::Program(context, source);
  try {
    program.build(vector < cl::Device > {device});
  }
  catch (cl::Error error) {
    if (error.err() == CL_BUILD_PROGRAM_FAILURE)
      cout << "Build log:" << endl << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;
    return 1;
  }

  kernel = cl::Kernel(program, "Convolution");

  return 0;
}

mBuffer CNeuron::convolve(const mBuffer inImgBuf) {
  int convImgWidth  = inImgBuf.width;
  int convImgHeight = inImgBuf.height;
  cl::Buffer convImgBuf = cl::Buffer(context, NULL, sizeof(cl_int) * 3 * convImgWidth * convImgHeight, (void *)NULL);

  kernel.setArg(0, sizeof(cl_mem), (void*)inImgBuf.data.get());
  kernel.setArg(1, sizeof(cl_int), &kernelWidth);
  kernel.setArg(2, sizeof(cl_int), &kernelHeight);
  kernel.setArg(3, sizeof(cl_mem), (void*)&kernelBuf);
  kernel.setArg(4, sizeof(cl_mem), (void*)&convImgBuf);
  
  commandQueue.enqueueNDRangeKernel(kernel, cl::NDRange(2), cl::NDRange(convImgHeight-1, convImgWidth), cl::NullRange);
  commandQueue.finish();
  
  mBuffer res = {
    make_shared<cl::Buffer>(convImgBuf),
    convImgWidth,
    convImgHeight
  };

  return res;
}

void CNeuron::setKernel(float *kernelData, int kernelWidth) {
  CNeuron::kernelWidth = kernelWidth;
  CNeuron::kernelHeight = kernelWidth;
  kernelBuf = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float) * kernelWidth * kernelHeight, (void*)kernelData);
}


PNeuron::PNeuron(float poolCoef, const cl::Context & context, const cl::Device & device, const cl::CommandQueue & commandQueue) :
  Neuron(context, device, commandQueue),
  poolCoef(poolCoef) {
  init();
}

PNeuron::PNeuron(const cl::Context & context, const cl::Device & device, const cl::CommandQueue & commandQueue) :
  Neuron(context, device, commandQueue) {
  init();
}

int PNeuron::init() {
  ifstream fin;
  fin.exceptions(ifstream::failbit | ifstream::badbit);
  try {
    fin.open(clPoolFileName);
    kernelSrc = string(istreambuf_iterator<char>(fin), (istreambuf_iterator<char>()));
    source = cl::Program::Sources(1, std::make_pair(kernelSrc.c_str(), kernelSrc.length() + 1));
    fin.close();
  }
  catch (ifstream::failure error) {
    cerr << "Exception opening/reading/closing 'MaxPooling.cl'" << endl;
    return 1;
  }

  program = cl::Program(context, source);

  try {
    program.build(vector < cl::Device > {device});
  }
  catch (cl::Error error) {
    if (error.err() == CL_BUILD_PROGRAM_FAILURE)
      cout << "Build log:" << endl << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;
    return 1;
  }

  kernel = cl::Kernel(program, "Pooling");

  return 0;
}


mBuffer PNeuron::pool(const mBuffer inImgBuf)
{
  int poolImgWidth = int(poolCoef * inImgBuf.width);
  int poolImgHeight = int(poolCoef * inImgBuf.height);
  cl::Buffer poolImgBuf = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(cl_int) * 3 * poolImgWidth * poolImgHeight, NULL);

  kernel.setArg(0, sizeof(cl_mem), (void*)inImgBuf.data.get());
  kernel.setArg(1, sizeof(cl_float), &poolCoef);
  kernel.setArg(2, sizeof(cl_mem), (void*)&poolImgBuf);

  commandQueue.enqueueNDRangeKernel(kernel, cl::NDRange(2), cl::NDRange(poolImgWidth, poolImgHeight), cl::NullRange);
  commandQueue.finish();

  mBuffer res;
  res.data = make_shared<cl::Buffer>(poolImgBuf);
  res.width = poolImgWidth;
  res.height = poolImgHeight;

  return res;
}

void PNeuron::setPoolCoef(float poolCoef) {
  PNeuron::poolCoef = poolCoef;
}



Layer::~Layer() { }

Layer::Layer() { }

void Layer::activate(vector<mBuffer> prevFeatureMaps, const cl::Context & context)
{
  cout << "Virtual activate hidden layer function called\n";
}


ILayer::ILayer(Mat inImage) :
  inImage(inImage) { }

void ILayer::activate(const cl::Context & context) {
  mBuffer image {
    make_shared<cl::Buffer>(cl::Buffer(context, CL_MEM_USE_HOST_PTR, sizeof(cl_int) * 3 * inImage.size().width * inImage.size().height, inImage.data)),
    inImage.size().width,
    inImage.size().height 
  };

  featureMaps.push_back(image);
}

  
OLayer::OLayer() : Layer() {}

void OLayer::activate(vector<mBuffer> prevFeatureMaps, const cl::Context & context) { }


CLayer::CLayer(vector<shared_ptr<CNeuron>> neurons) :
  neurons(neurons) { }

void CLayer::activate(vector<mBuffer> prevFeatureMaps, const cl::Context &context) {
  for (int j = 0; j < prevFeatureMaps.size(); j++) {
    for (int i = 0; i < neurons.size(); i++) {
      featureMaps.push_back(neurons[i].get()->convolve(prevFeatureMaps[j]));
    }
  }
  //cout << "CLayer done" << endl;
}


PLayer::PLayer(shared_ptr<PNeuron> neuron) :
  neuron(neuron) { }

void PLayer::activate(vector<mBuffer> prevFeatureMaps, const cl::Context & context) { 
  for (int j = 0; j < prevFeatureMaps.size(); j++) {
    featureMaps.push_back(neuron.get()->pool(prevFeatureMaps[j]));
  }
}
