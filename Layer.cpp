#include "Layer.h"

#include <fstream>
#include <iostream>

Neuron::Neuron(const cl::Context & context, const cl::Device & device, const cl::CommandQueue & commandQueue) :
  context(context),
  device(device),
  commandQueue(commandQueue) {
}

const vector<Mat> &Neuron::getFeatureMaps() {
  return featureMaps;
}

void Neuron::setFeatureMap(vector<Mat> featureMaps) {
  Neuron::featureMaps = featureMaps;
}


CNeuron::CNeuron(float *kernelData, int kernelWidth, const cl::Context &context, const cl::Device &device, const cl::CommandQueue &commandQueue) : 
  Neuron(context, device, commandQueue),
  kernelWidth(kernelWidth),
  kernelHeight(kernelWidth){
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

int CNeuron::convolve(const shared_ptr<cl::Buffer> inImgBuf, int inImgWidth, int inImgHeight) {
  int convImgWidth = inImgWidth - (kernelWidth - 1);
  int convImgHeight = inImgHeight - (kernelHeight - 1);
  cl::Buffer convImgBuf = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_uchar) * 3 * convImgWidth * convImgHeight, (void *)NULL);

  kernel.setArg(0, sizeof(cl_mem), (void*)inImgBuf.get());
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
Layer<NeuronType>::~Layer()
{
}

template<typename NeuronType>
Layer<NeuronType>::Layer() { }

template<typename NeuronType>
void Layer<NeuronType>::activate(vector<shared_ptr<Neuron>> prevNeurons, const cl::Context &context) {
  cout << "Virtual activate hidden layer function called\n";
}

template<typename NeuronType>
vector<shared_ptr<NeuronType>> Layer<NeuronType>::getNeurons() {
  return vector<shared_ptr<NeuronType>>(neurons);
}

template<typename NeuronType>
void Layer<NeuronType>::setNeurons(vector<shared_ptr<NeuronType>> neurons) {
  Layer::neurons = neurons;
}

ILayer::ILayer(char* path_) : Layer<Neuron>() {
  /*
  Mat frame = imread("input.png", CV_LOAD_IMAGE_COLOR);
  Mat img;
  cvtColor(frame, img, COLOR_BGR2GRAY);
  */
}

void ILayer::activate(vector<shared_ptr<Neuron>> prevNeurons, const cl::Context &context) {
  printf("ILayer done\n");
}

OLayer::OLayer() : Layer<Neuron>() {}

void OLayer::activate(vector<shared_ptr<Neuron>> prevNeurons, const cl::Context &context) {
  printf("OLayer done\n");
}

CLayer::CLayer(vector<shared_ptr<CNeuron>> neurons) : Layer<CNeuron>() {
  setNeurons(neurons);
}

void CLayer::activate(vector<shared_ptr<Neuron>> prevNeurons, const cl::Context &context) {

  for (int j = 0; j < prevNeurons.size(); j++) {
    const vector<Mat> &prevFmaps = prevNeurons[0].get()->getFeatureMaps();
    for (int k = 0; k < prevFmaps.size(); k++) {
      int inImgWidth = prevFmaps[k].size().width;
      int inImgHeight = prevFmaps[k].size().height;
      
      shared_ptr<cl::Buffer> inImgBuf = make_shared<cl::Buffer>(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_uchar) * 3 * inImgWidth * inImgHeight, (void*)prevFmaps[k].data);
      for (int i = 0; i < neurons.size(); i++) {
        neurons[i].get()->convolve(inImgBuf, inImgWidth, inImgHeight);
      }
    }
  }

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

PLayer::PLayer(vector<shared_ptr<PNeuron>> neurons) : Layer<PNeuron>() {
  setNeurons(neurons);
}

void
PLayer::activate(vector<shared_ptr<Neuron>> prevNeurons, const cl::Context &context) {
  //printf("PLayer done\n");
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