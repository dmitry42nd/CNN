#include "Layer.h"

#include <fstream>
#include <iostream>

const vector<Mat> &Neuron::getFeatureMaps() {
  return featureMaps;
}

CNeuron::CNeuron(float *kernelData, int kernelWidth, const cl::Context &context, const cl::Device &device, const cl::CommandQueue &commandQueue) :
  kernelWidth(kernelWidth),
  kernelHeight(kernelWidth),
  context(context),
  device(device),
  commandQueue(commandQueue) {
  setKernel(kernelData, kernelWidth);
  init();
}

CNeuron::CNeuron(const cl::Context &context, const cl::Device &device, const cl::CommandQueue &commandQueue) :
  context(context),
  device(device),
  commandQueue(commandQueue) {
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

int CNeuron::convolve(const cl::Buffer *inImgBuf, int inImgWidth, int inImgHeight) {
  int convImgWidth = inImgWidth - (kernelWidth - 1);
  int convImgHeight = inImgHeight - (kernelHeight - 1);
  cl::Buffer convImgBuf = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_uchar) * 3 * convImgWidth * convImgHeight, (void *)NULL);

  kernel.setArg(0, sizeof(cl_mem), (void*)inImgBuf);
  kernel.setArg(1, sizeof(cl_int), &kernelWidth);
  kernel.setArg(2, sizeof(cl_int), &kernelHeight);
  kernel.setArg(3, sizeof(cl_mem), (void*)&kernelBuf);
  kernel.setArg(4, sizeof(cl_mem), (void*)&convImgBuf);

  commandQueue.enqueueNDRangeKernel(kernel, cl::NDRange(2), cl::NDRange(convImgWidth, convImgHeight), cl::NullRange);
  commandQueue.finish();

  Mat convImage = Mat::zeros(Size(convImgWidth, convImgHeight), CV_8UC3);
  commandQueue.enqueueReadBuffer(convImgBuf, CL_TRUE, 0, sizeof(cl_uchar) * 3 * convImgWidth * convImgHeight, convImage.data);
  featureMaps.push_back(convImage);

  return 0;
}

void CNeuron::setKernel(float *kernelData, int kernelWidth) {
  CNeuron::kernelWidth = kernelWidth;
  CNeuron::kernelHeight = kernelWidth;
  kernelBuf = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float) * kernelWidth * kernelHeight, (void*)kernelData);
}


PNeuron::PNeuron(const cl::Context & context, const cl::Device & device, const cl::CommandQueue & commandQueue) :
  context(context),
  device(device),
  commandQueue(commandQueue) {
  init();
}

PNeuron::PNeuron(float poolCoef, const cl::Context & context, const cl::Device & device, const cl::CommandQueue & commandQueue) :
  poolCoef(poolCoef),
  context(context),
  device(device),
  commandQueue(commandQueue) {
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

int PNeuron::pool(const cl::Buffer * inImgBuf, int inImgWidth, int inImgHeight)
{
  int poolImgWidth = int(poolCoef * inImgWidth);
  int poolImgHeight = int(poolCoef * inImgHeight);
  cl::Buffer poolImgBuf = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_uchar) * 3 * poolImgWidth * poolImgHeight, NULL);

  kernel.setArg(0, sizeof(cl_mem), (void*)inImgBuf);
  kernel.setArg(1, sizeof(cl_float), &poolCoef);
  kernel.setArg(2, sizeof(cl_mem), (void*)&poolImgBuf);

  commandQueue.enqueueNDRangeKernel(kernel, cl::NDRange(2), cl::NDRange(poolImgWidth, poolImgHeight), cl::NullRange);
  commandQueue.finish();

  Mat poolImage = Mat::zeros(Size(poolImgWidth, poolImgHeight), CV_8UC3);
  commandQueue.enqueueReadBuffer(poolImgBuf, CL_TRUE, 0, sizeof(cl_uchar) * 3 * poolImgWidth * poolImgHeight, poolImage.data);
  featureMaps.push_back(poolImage);

  return 0;
}

void PNeuron::setPoolCoef(float poolCoef) {
  PNeuron::poolCoef = poolCoef;
}





template<typename NeuronType>
Layer<NeuronType>::Layer() { }

template<typename NeuronType>
void Layer<NeuronType>::activate(Layer<NeuronType> * prevLayer_) {
  cout << "Virtual activate hidden layer function called\n";
}

template<typename NeuronType>
shared_ptr<vector<NeuronType>> Layer<NeuronType>::getNeurons() {
  return neurons;
}

template<typename NeuronType>
void Layer<NeuronType>::setNeurons(shared_ptr<vector<NeuronType>> neurons) {
  Layer::neurons = neurons;
}

ILayer::ILayer(char* path_) {
  /*
  Mat frame = imread("input.png", CV_LOAD_IMAGE_COLOR);
  Mat img;
  cvtColor(frame, img, COLOR_BGR2GRAY);
  */
}

void ILayer::activate(Layer<Neuron> *prevLayer_) {
  printf("ILayer done\n");
}

shared_ptr<vector<Neuron>> ILayer::getNeurons() {
  return neurons;
}

OLayer::OLayer() {}

void OLayer::activate(Layer<Neuron> *prevLayer_) {
  printf("OLayer done\n");
}

shared_ptr<vector<Neuron>> OLayer::getNeurons() {
  return neurons;
}

CLayer::CLayer(shared_ptr<vector<CNeuron>> neurons) {
  setNeurons(neurons);
}

void CLayer::activate(Layer *prevLayer_) {
  //shared_ptr<vector<Neuron>> prevNeurons = prevLayer_->getNeurons();
  //const vector<Neuron> &neurons = prevLayer_->getNeurons();
  /*
  foreach neurons {
  foreach prevNeurons, i {
  foreach prevNeuronsFmaps;
  neuron.fmaps[i] = neuron.convolve(prevNeuronsFmap);
  }
  }
  }
  */
  cout << "CLayer done\n";
}

shared_ptr<vector<CNeuron>> CLayer::getNeurons() {
  return neurons;
}

PLayer::PLayer(shared_ptr<vector<PNeuron>> neurons) {
  setNeurons(neurons);
}

void
PLayer::activate(Layer *prevLayer_) {
  //printf("SLayer done\n");
  /*
  foreach neurons {
  foreach prevNeurons {
  foreach prevNeuronsFmaps {
  neuron.pool(prevNeuronsFmap);
  }
  }
  }
  */
}

shared_ptr<vector<PNeuron>> PLayer::getNeurons() {
  return neurons;
}