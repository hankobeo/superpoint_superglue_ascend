

#pragma once

#include "basic.h"
#include <vector>
#include <sys/time.h>
#include <algorithm>
#include "opencv2/opencv.hpp"

using namespace std;

class superpoint
{
private:
    uint32_t g_modelWidth_;   // The input width required by the model
    uint32_t g_modelHeight_;  // The model requires high input
    const char *g_modelPath_; // Offline model file path
    basic *Superpoint_Instant;
    aclmdlIODims *out_dims;    //输出张量内容
    aclmdlIODims *in_dims;
   

public:

    superpoint(const char *modelPath, uint32_t modelWidth, uint32_t modelHeight);
    ~superpoint();

    Result Init();
    // Result Preprocess(ImgMat frame);
    
    Result Preprocess(cv::Mat image);
    Result Inference();
    std::vector<float> Postprocess();
    Result UnInit();

    void find_high_score_index(std::vector<float> &scores, std::vector<std::vector<int>> &keypoints, int h, int w, float threshold);
    void remove_borders(std::vector<std::vector<int>> &keypoints, std::vector<float> &scores, int border, int height, int width);
    std::vector<size_t> sort_indexes(std::vector<float> &data);
    void top_k_keypoints(std::vector<std::vector<int>> &keypoints, std::vector<float> &scores, int k);

    void sample_descriptors(std::vector<std::vector<int>> &keypoints, float *descriptors,
                                    std::vector<std::vector<double>> &dest_descriptors, int dim, int h, int w, int s);
    void normalize_keypoints(const std::vector<std::vector<int>> &keypoints, std::vector<std::vector<double>> &keypoints_norm,
                    int h, int w, int s);
    void grid_sample(const float *input, std::vector<std::vector<double>> &grid,
                 std::vector<std::vector<double>> &output, int dim, int h, int w);
    void normalize_descriptors(std::vector<std::vector<double>> &dest_descriptors);
    void visualization(const std::string &image_name, const cv::Mat &image);

    std::vector<std::vector<int>> keypoints_;
    std::vector<std::vector<double>> descriptors_;

};
