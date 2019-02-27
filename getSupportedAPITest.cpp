/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <algorithm>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <map>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <unistd.h> // For ::getopt
#include <string>
#include <NvInfer.h>
#include "NvOnnxParser.h"
#include "onnx_utils.hpp"
#include "common.hpp"

using std::cout;
using std::cerr;
using std::endl;

#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cout << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

void print_usage() {
  cout << "This program will determine whether or not an ONNX model is compatible with TensorRT. " 
       << "If it isn't, a list of supported subgraphs and unsupported operations will be printed." << endl;
  cout << "Usage: getSupportedAPITest -m onnx_model.pb" << endl;
  cout << "Optional argument: -e TRT_engine" << endl;
}

void printSubGraphs(SubGraphCollection_t& subGraphs, ::ONNX_NAMESPACE::ModelProto onnx_model)
{
    if (subGraphs.size() > 1)
    {
        cout << "The model contains unsupported Nodes. It has been partitioned to a set of supported subGraphs." << endl;
        cout << "There are "<< subGraphs.size() << " supported subGraphs: " << endl;
    }
    else 
    {
        cout << "The model is fully supported by TensorRT. Printing the parsed graph:" << endl;
    }

    for (auto subGraph: subGraphs) 
    {
        cout << "\t{";
        for (auto idx: subGraph) cout << "\t" << idx << "," <<onnx_model.graph().node(idx).op_type();
        cout << "\t}"<<endl;
    }
}

static std::map<std::string, std::string> getInputOutputNames(const nvinfer1::ICudaEngine& trt_engine)
{
    int nbindings = trt_engine.getNbBindings();
    assert(nbindings == 2);
    std::map<std::string, std::string> tmp;
    for (int b = 0; b < nbindings; ++b)
    {
        if (trt_engine.bindingIsInput(b))
        {
            tmp["input"] = trt_engine.getBindingName(b);
        }
        else
        {
            tmp["output"] = trt_engine.getBindingName(b);
        }
    }
    return tmp;
}

static int volume(nvinfer1::Dims dims)
{
    return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int>());
}

static void doInference(nvinfer1::ICudaEngine& trt_engine,
                void ** buffers,
                std::vector <float>& input,
                std::vector <float>& output)
{
    // First dim = batch size = 3.
    int batchSize = 3;
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    // Get input and output binding names
    std::map<std::string, std::string> InOut = getInputOutputNames(trt_engine);
    const int inputIndex = trt_engine.getBindingIndex(InOut["input"].c_str()), outputIndex = trt_engine.getBindingIndex(InOut["output"].c_str());

    cout << "input index " << inputIndex << endl;
    cout << "output index " << outputIndex << endl;

    nvinfer1::Dims inp_dims = trt_engine.getBindingDimensions(inputIndex);
    nvinfer1::Dims output_dims = trt_engine.getBindingDimensions(outputIndex);

    size_t inputDim = volume(inp_dims);
    size_t outputDim = volume(output_dims);
    output.resize(outputDim*batchSize);

    // Get total memory size of input and output 
    size_t input_size = batchSize * inputDim * sizeof(float);
    size_t output_size = batchSize * outputDim * sizeof(float);
    CHECK(cudaMalloc(&buffers[inputIndex], input_size));
    CHECK(cudaMalloc(&buffers[outputIndex], output_size));

    CHECK(cudaStreamCreate(&stream));

    // Copy the input data over to the input buffer
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input.data(), input_size, cudaMemcpyHostToDevice, stream));

    nvinfer1::IExecutionContext* context = trt_engine.createExecutionContext();
    // Enqueue to context, with batchSize = 3.
    context->enqueue(batchSize, buffers, stream, nullptr);

    // Copy output data bck to the output buffer
    CHECK(cudaMemcpyAsync(output.data(), buffers[outputIndex], output_size, cudaMemcpyDeviceToHost, stream));

    CHECK(cudaStreamSynchronize(stream));

    CHECK(cudaStreamDestroy(stream));

    CHECK(cudaFree(buffers[inputIndex]));

    CHECK(cudaFree(buffers[outputIndex]));

    context->destroy();
}


int main(int argc, char* argv[]) {

    GOOGLE_PROTOBUF_VERIFY_VERSION;

    std::string engine_filename;
    std::string text_filename;
    std::string full_text_filename;
    std::string onnx_filename;
    int c;
    size_t max_batch_size = 32;
    size_t max_workspace_size = 16 << 20;
    int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;
    while ((c = getopt (argc, argv, "m:e:")) != -1)
    {
        switch(c)
        {
            case 'm':
                    onnx_filename = optarg;
                    break;
            case 'e':
                    engine_filename = optarg;
                    break;
        }
    }

    if (onnx_filename.empty())
    {
        print_usage();
        return -1;
    }

    common::TRT_Logger trt_logger((nvinfer1::ILogger::Severity)verbosity);
    auto trt_builder = common::infer_object(nvinfer1::createInferBuilder(trt_logger));
    auto trt_network = common::infer_object(trt_builder->createNetwork());
    auto trt_parser  = common::infer_object(nvonnxparser::createParser(*trt_network, trt_logger));

    cout << "Parsing model: " << onnx_filename << endl;
    
    std::ifstream onnx_file(onnx_filename.c_str(),
                            std::ios::binary | std::ios::ate);
    std::streamsize file_size = onnx_file.tellg();
    onnx_file.seekg(0, std::ios::beg);
    std::vector<char> onnx_buf(file_size);

    if( !onnx_file.read(onnx_buf.data(), onnx_buf.size()) ) {
        cerr << "ERROR: Failed to read from file " << onnx_filename << endl;
        return 1;
    }

    ::ONNX_NAMESPACE::ModelProto onnx_model;
    common::ParseFromFile_WAR(&onnx_model, onnx_filename.c_str());

    SubGraphCollection_t SubGraphCollection;

    if (!trt_parser->supportsModel(onnx_buf.data(), onnx_buf.size(), SubGraphCollection))
    {
        cout << "Model cannot be fully parsed by TensorRT!" << endl;
        printSubGraphs(SubGraphCollection, onnx_model);
        return -1;
    }

    printSubGraphs(SubGraphCollection, onnx_model);
    
    trt_builder->setMaxBatchSize(max_batch_size);
    trt_builder->setMaxWorkspaceSize(max_workspace_size);
    trt_parser->parse(onnx_buf.data(), onnx_buf.size());

    cout << "input name: " << trt_network->getInput(0)->getName() << endl;
    cout << "output name: " << trt_network->getOutput(0)->getName() << endl;
    cout << "num layers: " << trt_network->getNbLayers() << endl;
    cout << "outputs: " << trt_network->getNbOutputs() << endl;
    auto trt_engine = trt_builder->buildCudaEngine(*trt_network);
    void* buffers[2];
    std::vector<float>input;
    // Create input data for abs.onnx. 3x4x5 = 60 total elements.
    for (size_t i = 0; i < 60; i++)
    {
        input.push_back(i-30.0);
    }
    std::vector<float>output;

    // Do inference
    doInference(*trt_engine, buffers, input, output);
    for (size_t i = 0; i < 60; i++)
    {
        if (abs(input[i]) != output[i]) 
        {
            cout << "Failed" << endl;
            cout << input[i] << "\t" << output[i] << endl;
        }   
    }
    cout << "Done inference on 3D input" << endl;
    
    if( verbosity >= (int)nvinfer1::ILogger::Severity::kWARNING ) {
        cout << "All done" << endl;
    }
    return 0;
}
