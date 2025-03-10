#include "basic.h"

basic::basic(const char* modelPath)
    :g_deviceId_(0)
{

    omModelPath = modelPath;
}

basic::~basic()
{
}


Result basic::Init_atlas()
{
    // ACL init
    const char *aclConfigPath = nullptr;
    aclError ret = aclInit(aclConfigPath);
    if (ret == 100002) {
        std::cout<<"acl init already"<<std::endl;
    }
    else if(ret)
    {
        std::cout<<"acl get run mode failed"<<std::endl;
        return FAILED;
    }
    
    // open device
    ret = aclrtSetDevice(g_deviceId_);
    if (ret) {
        std::cout<<"Acl open device failed"<<std::endl;
        return FAILED;
    }

    // create context
    ret = aclrtCreateContext(&g_context_, g_deviceId_);
    if (ret) {
        std::cout<<"acl create context failed"<<std::endl;
        return FAILED;
    }

    // create stream
    ret = aclrtCreateStream(&g_stream_);
    if (ret) {
        std::cout<<"acl create stream failed"<<std::endl;
        return FAILED;
    }

    // Gets whether the current application is running on host or Device
    ret = aclrtGetRunMode(&g_runMode_);
    if (ret) {
        std::cout<<"acl get run mode failed"<<std::endl;
        return FAILED;
    }

    ret = aclmdlLoadFromFile(omModelPath, &g_modelId_);
    if(ret){
        std::cout<<"loadding model error, model name: " << omModelPath << std::endl;
        return FAILED;
    }

    g_modelDesc_ = aclmdlCreateDesc();
    if (g_modelDesc_ == nullptr) {
        std::cout<<"create model description failed"<<std::endl;
        return FAILED;
    }

    ret = aclmdlGetDesc(g_modelDesc_, g_modelId_);
    if (ret) {
        std::cout<<"get model description failed"<<std::endl;
        return FAILED;
    }

    return SUCCESS;
}


//创造输入数据
Result basic::CreateInput() {

    // aclError aclRet = aclrtMalloc(&g_imageDataBuf_, g_imageDataSize_, ACL_MEM_MALLOC_HUGE_FIRST);

    g_input_ = aclmdlCreateDataset();
    if (g_input_ == nullptr) {
        std::cout<<"can't create dataset, create input failed"<<std::endl;
        return FAILED;
    }
    
    inputSize = aclmdlGetNumInputs(g_modelDesc_);

    for (size_t i = 0; i < inputSize; ++i) {
        size_t buffer_size = aclmdlGetInputSizeByIndex(g_modelDesc_, i);

        void *inputBuffer = nullptr;
        aclError ret = aclrtMalloc(&inputBuffer, buffer_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret) {
            std::cout<<"can't malloc buffer, create input failed"<<std::endl;
            return FAILED;
        }

        aclDataBuffer* inputData = aclCreateDataBuffer(inputBuffer, buffer_size);
        if (inputData == nullptr) {
            std::cout<<"can't create data buffer, create input failed"<<std::endl;
            return FAILED;
        }

        ret = aclmdlAddDatasetBuffer(g_input_, inputData);
        if (inputData == nullptr) {
            std::cout<<"can't add data buffer, create input failed"<<std::endl;
            aclDestroyDataBuffer(inputData);
            inputData = nullptr;
            return FAILED;
        }

    }

    return SUCCESS;
}

Result basic::GetInputShape(aclmdlIODims* in_dims) 
{   
    printf("输入个数为：%d\n", inputSize);
    for(int i = 0; i < inputSize; i++){
        // 通过传入的 out_dims 指针获取模型的输出维度
        int ret = aclmdlGetInputDims(g_modelDesc_, i, in_dims);  // 获取模型输出的维度信息
        if (ret) {
            std::cerr << "Failed to get insput dims for index, " << i << std::endl;
            return FAILED;
        }
        ret = aclmdlGetInputDataType(g_modelDesc_, i);

        std::cout <<"Intput "<<i<<" datatype = "<< ret<<", ";

        for (size_t j = 0; j < in_dims->dimCount; j++){
            std::cout << in_dims->dims[j];
            if (j < in_dims->dimCount - 1) {
                std::cout << "x";
            }
        }
        std::cout << std::endl;
    }

    return SUCCESS;
}

Result basic::GetOutputShape(aclmdlIODims* out_dims) 
{   
    printf("输出个数为：%d\n", outputSize);
    for(int i = 0; i < outputSize; i++){
        // 通过传入的 out_dims 指针获取模型的输出维度
        int ret = aclmdlGetCurOutputDims(g_modelDesc_, i, out_dims);  // 获取模型输出的维度信息
        if (ret) {
            std::cerr << "Failed to get Output Dims " << i << std::endl;
            return FAILED;
        }

        ret = aclmdlGetOutputDataType(g_modelDesc_, i);

        std::cout <<"Output "<<i<<" datatype = "<< ret<<", ";
    
        for (size_t j = 0; j < out_dims->dimCount; j++) {
            std::cout << out_dims->dims[j];
            if (j < out_dims->dimCount - 1) {
                std::cout << "x";
            }
        }
        std::cout << std::endl;

    }


    return SUCCESS;
}


//创造输出数据
Result basic::CreateOutput() {

    if (g_modelDesc_ == nullptr) {
        std::cout<<"no model description, create ouput failed"<<std::endl;
        return FAILED;
    }


    g_output_ = aclmdlCreateDataset();
    if (g_output_ == nullptr) {
        std::cout<<"can't create dataset, create output failed"<<std::endl;
        return FAILED;
    }

    outputSize = aclmdlGetNumOutputs(g_modelDesc_);
    for (size_t i = 0; i < outputSize; ++i) {
        size_t buffer_size = aclmdlGetOutputSizeByIndex(g_modelDesc_, i);

        void *outputBuffer = nullptr;
        aclError ret = aclrtMalloc(&outputBuffer, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret) {
            std::cout<<"can't malloc buffer, create output failed"<<std::endl;
            return FAILED;
        }

        aclDataBuffer* outputData = aclCreateDataBuffer(outputBuffer, buffer_size);
        if (ret) {
            std::cout<<"can't create data buffer, create output failed"<<std::endl;
            aclrtFree(outputBuffer);
            return FAILED;
        }

        ret = aclmdlAddDatasetBuffer(g_output_, outputData);
        if (ret) {
            std::cout<<"can't add data buffer, create output failed"<<std::endl;
            aclrtFree(outputBuffer);
            aclDestroyDataBuffer(outputData);
            return FAILED;
        }
    }

    return SUCCESS;
}

Result basic::SetProcessInputItem(void* data, uint32_t idx){
    aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(g_input_, idx);
    if (dataBuffer == nullptr) {
        std::cout<<"Get dataset buffer from model failed "<<std::endl;
        return FAILED;
    }

    void* dataBufferDev = aclGetDataBufferAddr(dataBuffer);
    if (dataBufferDev == nullptr) {
        std::cout<<"Model inference output failed" <<std::endl;
        return FAILED;
    }

    size_t bufferSize = aclGetDataBufferSizeV2(dataBuffer);
    if (bufferSize == 0) {
        std::cout<<"Model inference output is 0" <<std::endl;
        return FAILED;
    }
    int ret = 0;
    if (g_runMode_ == ACL_HOST)
        ret = aclrtMemcpy(dataBufferDev, bufferSize, data, bufferSize, ACL_MEMCPY_HOST_TO_DEVICE);
    else
        ret = aclrtMemcpy(dataBufferDev, bufferSize, data, bufferSize, ACL_MEMCPY_DEVICE_TO_DEVICE);

    if (ret) {
        std::cout<<"Copy process data to device failed."<<std::endl;
        return FAILED;
    }

    return SUCCESS;
}



void* basic::GetInferenceOutputItem(uint32_t idx)
{
    aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(g_output_, idx);
    if (dataBuffer == nullptr) {
        std::cout<<"Get dataset buffer from model failed "<<std::endl;
        return nullptr;
    }

    void* dataBufferDev = aclGetDataBufferAddr(dataBuffer);
    if (dataBufferDev == nullptr) {
        std::cout<<"Model inference output failed" <<std::endl;
        return nullptr;
    }

    size_t bufferSize = aclGetDataBufferSizeV2(dataBuffer);
    if (bufferSize == 0) {
        std::cout<<"Model inference output is 0" <<std::endl;
        return nullptr;
    }

    void* data = nullptr;
    if (g_runMode_ == ACL_HOST) {
        data = CopyDataDeviceToLocal(dataBufferDev, bufferSize);
        if (data == nullptr) {
            std::cout<<"Copy inference output to host failed"<<std::endl;
            return nullptr;
        }
    } else {
        data = dataBufferDev;
        std::cout<<"--RC model--"<<std::endl;
    }

    // itemDataSize = bufferSize;
    return data;
}

void* basic::CopyDataDeviceToLocal(void* deviceData, uint32_t dataSize)
{
    // uint8_t* buffer = new uint8_t[dataSize];
    // if (buffer == nullptr) {
    //     std::cout<<"New malloc memory failed"<<std::endl;
    //     return nullptr;
    // }

    // aclError aclRet = aclrtMemcpy(buffer, dataSize, deviceData, dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    // if (aclRet) {
    //     std::cout<<"Copy device data to local failed" << std::endl;
    //     delete[](buffer);
    //     return nullptr;
    // }

    // return (void*)buffer;

    void* buffer = nullptr;
    aclError ret = aclrtMallocHost(&buffer, dataSize);
    if (ret != ACL_SUCCESS) {
	cout << "aclrtMallocHost failed, result code is " << ret << endl;
    }
    
    aclError aclRet = aclrtMemcpy(buffer, dataSize, deviceData, dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_SUCCESS) {
        std::cout<<"Copy device data to local failed"<<std::endl;;
        //delete[](buffer);
	    (void)aclrtFreeHost(buffer);
        return nullptr;
    }

    return (void*)buffer;
}

Result basic::inference() {
    int ret = aclrtSetCurrentContext(g_context_);
    ret = aclmdlExecute(g_modelId_, g_input_, g_output_);
    if(ret) {
        std::cout<<"Model inference failed"<<std::endl;
        return FAILED;
    }

    return SUCCESS;
}

void basic::DestroyResource() {

    aclrtFree(g_imageDataBuf_);
    aclError ret = aclmdlUnload(g_modelId_);
    // if (CLOCK_REALTIME_ALARM) {
    //     std::cout<<"unload model failed"<<std::endl;
    // }

    DestroyDesc();
    DestroyInput();
    DestroyOutput();

    ret = aclrtResetDevice(g_deviceId_);
    if (ret) {
        std::cout<<"reset device failed"<<std::endl;
    }

    ret = aclFinalize();
    if (ret) {
        std::cout<<"finalize acl failed"<<std::endl;
    }
    std::cout<<"end to finalize acl"<<std::endl;
    
}   


void basic::DestroyDesc()
{
    if (g_modelDesc_ != nullptr) {
        (void)aclmdlDestroyDesc(g_modelDesc_);
        g_modelDesc_ = nullptr;
    }
}

void basic::DestroyInput()
{
    if (g_input_ == nullptr) {
        return;
    }

    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(g_input_); ++i) {
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(g_input_, i);
        aclDestroyDataBuffer(dataBuffer);
    }
    aclmdlDestroyDataset(g_input_);
    g_input_ = nullptr;
}

void basic::DestroyOutput()
{
    if (g_output_ == nullptr) {
        return;
    }

    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(g_output_); ++i) {
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(g_output_, i);
        void* data = aclGetDataBufferAddr(dataBuffer);
        (void)aclrtFree(data);
        (void)aclDestroyDataBuffer(dataBuffer);
    }

    (void)aclmdlDestroyDataset(g_output_);
    g_output_ = nullptr;
}


