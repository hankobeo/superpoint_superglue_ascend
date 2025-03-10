

#pragma once

#include "basic.h"
#include <vector>
#include <sys/time.h>
#include <algorithm>
#include "opencv2/opencv.hpp"

using namespace std;


class superglue
{
private:
    uint32_t g_modelWidth_;   // The input width required by the model
    uint32_t g_modelHeight_;  // The model requires high input
    const char *g_modelPath_; // Offline model file path
    basic *Superglue_Instant;
    aclmdlIODims *out_dims;    //输出张量内容
    aclmdlIODims *in_dims;


public:

    superglue(const char *modelPath, uint32_t modelWidth, uint32_t modelHeight);
    ~superglue();

    Result Init();
    // Result Preprocess(ImgMat frame);

    Result Preprocess(std::vector<std::vector<int>> keypoints0, 
                            std::vector<float> scores0,
                            std::vector<std::vector<double>> descriptors0,
                            std::vector<std::vector<int>> keypoints1, 
                            std::vector<float> scores1,
                            std::vector<std::vector<double>> descriptors1);

        
 
    Result Inference();
    Result Postprocess(std::vector<std::vector<int>> keypoints0, std::vector<std::vector<int>> keypoints1);
    Result Postprocess(std::vector<std::vector<int>> keypoints0, std::vector<std::vector<int>> keypoints1, cv::Mat img1, cv::Mat img2);
    Result UnInit();

    float * Preprocess_keypoints(std::vector<std::vector<int>> &keypoints);
    float * Preprocess_scores(std::vector<float> scores);
    float * Preprocess_descriptors(std::vector<std::vector<double>> descriptors);
    // void normalize_keypoints(std::vector<std::vector<int>> &keypoints, int width, int height);
    void normalize_keypoints(std::vector<std::vector<float>>& keypoints, int width, int height);
};
