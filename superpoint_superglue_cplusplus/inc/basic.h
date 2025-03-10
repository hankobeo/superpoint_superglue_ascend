#pragma once
#include <iostream>
#include "acl/acl.h"
#include "type_api.h"

using namespace std;

class basic
{
private:
    /* data */
    uint32_t g_modelId_;
    int32_t g_deviceId_;  // Device ID, default is 0

    
    aclmdlDesc *g_modelDesc_;
    aclrtContext g_context_;
    aclrtStream g_stream_;
    aclmdlDataset *g_input_;
    aclmdlDataset *g_output_;

    
    
    const char* omModelPath;

public:
    basic(const char* modelPath);
    ~basic();
    
    void*  g_imageDataBuf_;      // Model input data cache
    uint32_t g_imageDataSize_; // Model input data size
    aclrtRunMode g_runMode_;

    size_t inputSize;
    size_t outputSize;
  
public:
    Result Init_atlas();

    Result CreateInput();
    Result CreateOutput();
    Result GetOutputShape(aclmdlIODims* out_dims);
    Result GetInputShape(aclmdlIODims* in_dims);

    Result inference();

    Result SetProcessInputItem(void* data, uint32_t idx);
    void* GetInferenceOutputItem(uint32_t idx);

    void* CopyDataDeviceToLocal(void* deviceData, uint32_t dataSize);
    Result GetOutputShape(aclmdlIODims* out_dims, int index);
    
    void DestroyResource();
    void DestroyDesc();
    void DestroyInput();
    void DestroyOutput();
};

