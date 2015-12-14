#include "Layer.h"

#include <fstream>
#include <iostream>

Neuron::Neuron(const cl::Context & context, const cl::Device & device, const cl::CommandQueue & commandQueue) :
  context(context),
  device(device),
  commandQueue(commandQueue) {
}


CNeuron::CNeuron(vector<float*>kernelsData, int kernelWidth, const cl::Context &context, const cl::Device &device, const cl::CommandQueue &commandQueue) : 
  Neuron(context, device, commandQueue),
  kernelWidth(kernelWidth) {
  setKernels(kernelsData, kernelWidth);
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

shared_ptr<cl::Buffer> CNeuron::convolve(const FeatureMaps inFMaps) {
  int convImgWidth  = inFMaps.width;
  int convImgHeight = inFMaps.height;

  //just to init buffer by zeros
  cl_float *zeros = (cl_float *)calloc(3 * convImgWidth * convImgHeight, sizeof(cl_float));
  cl::Buffer convImgBuf = cl::Buffer(context, CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * 3 * convImgWidth * convImgHeight, (void *)zeros);

  kernel.setArg(1, sizeof(cl_int), &kernelWidth);
  kernel.setArg(3, sizeof(cl_mem), (void*)&convImgBuf);
  
  for (size_t j = 0; j < inFMaps.buffers.size(); j++) {
    kernelBuf = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float) * kernelWidth * kernelWidth, (void*)kernelsData[j]);
    kernel.setArg(0, sizeof(cl_mem), (void*)inFMaps.buffers[j].get());
    kernel.setArg(2, sizeof(cl_mem), (void*)&kernelBuf);
    commandQueue.enqueueNDRangeKernel(kernel, cl::NDRange(2), cl::NDRange(convImgWidth, convImgHeight), cl::NullRange);
    commandQueue.finish();
  }

  free(zeros);
  return make_shared<cl::Buffer>(convImgBuf);
}

void CNeuron::setKernels(vector<float*>kernelsData, int kernelWidth) {
  CNeuron::kernelWidth  = kernelWidth;
  CNeuron::kernelsData = kernelsData;
}


PNeuron::PNeuron(float poolBias, const cl::Context & context, const cl::Device & device, const cl::CommandQueue & commandQueue) :
  Neuron(context, device, commandQueue),
  poolBias(poolBias){
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

shared_ptr<cl::Buffer> PNeuron::pool(const shared_ptr<cl::Buffer> buffer, int outWidth, int outHeight, float poolCoef)
{
  int poolImgWidth = outWidth;
  int poolImgHeight = outHeight;

  kernel.setArg(1, sizeof(cl_float), &poolCoef);
  kernel.setArg(2, sizeof(cl_float), &poolBias);

  cl::Buffer poolImgBuf = cl::Buffer(context, NULL, sizeof(cl_float) * 3 * poolImgWidth * poolImgHeight, NULL);

  kernel.setArg(0, sizeof(cl_mem), (void*)buffer.get());
  kernel.setArg(3, sizeof(cl_mem), (void*)&poolImgBuf);

  commandQueue.enqueueNDRangeKernel(kernel, cl::NDRange(2), cl::NDRange(poolImgWidth, poolImgHeight), cl::NullRange);
  commandQueue.finish();

  return make_shared<cl::Buffer>(poolImgBuf);
}

void PNeuron::setPoolCoef(float poolBias) {
  PNeuron::poolBias = poolBias;
}



Layer::~Layer() {}
Layer::Layer() {}


ILayer::ILayer() {}

void ILayer::activate(Mat inImage, const cl::Context &context) {
  featureMaps.buffers.push_back(make_shared<cl::Buffer>(cl::Buffer(context, CL_MEM_USE_HOST_PTR, sizeof(cl_float) * 3 * inImage.size().width * inImage.size().height, inImage.data)));
  featureMaps.width  = inImage.size().width;
  featureMaps.height = inImage.size().height;
}

  
OLayer::OLayer() {}

void OLayer::activate(FeatureMaps prevFeatureMaps) { }


HiddenLayer::~HiddenLayer() {}
HiddenLayer::HiddenLayer() {}

void HiddenLayer::activate(FeatureMaps prevFeatureMaps)
{
  cout << "Virtual method of base class called. This shouldn't happen.";
}

CLayer::CLayer(vector<shared_ptr<CNeuron>> neurons) :
  neurons(neurons) { }

void CLayer::activate(FeatureMaps prevFeatureMaps) {
  featureMaps.width  = prevFeatureMaps.width;
  featureMaps.height = prevFeatureMaps.height;

  for (size_t i = 0; i < neurons.size(); i++) {
    featureMaps.buffers.push_back(neurons[i].get()->convolve(prevFeatureMaps));
  }
}


PLayer::PLayer(vector<shared_ptr<PNeuron>> neurons, float poolCoef) :
  neurons(neurons),
  poolCoef(poolCoef) { }

void PLayer::activate(FeatureMaps prevFeatureMaps) {
  featureMaps.width = static_cast<int>(prevFeatureMaps.width*poolCoef);
  featureMaps.height = static_cast<int>(prevFeatureMaps.height*poolCoef);

  for (size_t i = 0; i < neurons.size(); i++) {
    featureMaps.buffers.push_back(neurons[i].get()->pool(prevFeatureMaps.buffers[i], featureMaps.width, featureMaps.height, poolCoef));
  }
}