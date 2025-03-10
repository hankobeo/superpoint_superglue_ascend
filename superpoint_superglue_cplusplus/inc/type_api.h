#pragma once
#include <iostream>
#include <vector>
#include <unistd.h>
#include <stdlib.h>

using namespace std;

#define RGBU8_IMAGE_SIZE(width, height) ((width) * (height) * 3)
#define RGBAU8_IMAGE_SIZE(width, height) ((width) * (height) * 4)

#define NV12_IMAGE_SIZE(width, height) ((width) * (height) * 3 / 2)
#define RGBF32_IMAGE_SIZE(width, height) ((width) * (height) * 3 * 4)



//echo $AI_ALGO_PROFILE_PRINT 查看变量的值
//export -p  查看所有的环境变量

//export AI_ALGO_PROFILE_PRINT=1        打开耗时日志
//unset AI_ALGO_PROFILE_PRINT           关闭耗时日志    清理AI_ALGO_PROFILE_PRINT环境变量

// export AI_ALGO_DEBUG_PRINT=1         打开检测类别日志
// unset AI_ALGO_DEBUG_PRINT            关闭检测类别日志

// #define AI_PROFILE (getenv("AI_ALGO_PROFILE_PRINT")==NULL? 0:1)
// #define PROFILE_LOG(format, ...) if(AI_PROFILE) printf("\n[PROFILE:%s:%d->%s] " format, __FILE__, __LINE__, __func__, ##__VA_ARGS__)

// #define AI_DEBUG (getenv("AI_ALGO_DEBUG_PRINT")==NULL? 0:1)
// #define DEBUG_LOG(format, ...)   if(AI_DEBUG) printf("\n[DEBUG:%s:%d->%s] " format, __FILE__, __LINE__, __func__, ##__VA_ARGS__)

#define PROFILE_LOG(format, ...) if(access("./AI_PROFILE", F_OK) == 0) printf("\n[PROFILE:%s:%d->%s] " format, __FILE__, __LINE__, __func__, ##__VA_ARGS__)
#define DEBUG_LOG(format, ...) if(access("./AI_DEBUG", F_OK) == 0) printf("\n[DEBUG:%s:%d->%s] " format, __FILE__, __LINE__, __func__, ##__VA_ARGS__)

typedef enum  {
    SUCCESS = 0,
    FAILED = 1
}Result;

typedef struct  {
    float x;
    float y;
    float w;
    float h;
    float score;
    int classIndex;
    // int index; // index of output buffer
} BBox;


typedef struct 
{   
    std::vector<int>            stride;

    std::vector<int>            anchors_VL;
    std::vector<int>            anchors_IR;

    int                         netWidth_VL;
    int                         netHeight_VL;
    int                         netWidth_IR;
    int                         netHeight_IR;

    float                       confThresh;
    float                       iouThresh;

    int                         VL_cell_size;
    int                         VL_ojb_class_num;

    int                         IR_cell_size;
    int                         IR_ojb_class_num;

    int                         detect_num;

} ConfigData;
