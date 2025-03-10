#include "superpoint.h"
#include <numeric>

using namespace std;




superpoint::superpoint(const char* modelPath, uint32_t modelWidth,
                           uint32_t modelHeight)
    :g_modelWidth_(modelWidth), g_modelHeight_(modelHeight)
{
    g_modelPath_ = modelPath;
    out_dims = (aclmdlIODims*)malloc(sizeof(aclmdlIODims));
    in_dims = (aclmdlIODims*)malloc(sizeof(aclmdlIODims));
}

superpoint::~superpoint()
{
    if (out_dims) {
        free(out_dims);
        out_dims = nullptr;
    }

    if (in_dims) {
        free(in_dims);
        in_dims = nullptr;
    }
}


Result superpoint::Init() {

    Superpoint_Instant = new basic(g_modelPath_);

    Result ret = Superpoint_Instant->Init_atlas();
    ret = Superpoint_Instant->CreateInput();
    ret = Superpoint_Instant->CreateOutput();

    ret = Superpoint_Instant->GetInputShape(in_dims);
    ret = Superpoint_Instant->GetOutputShape(out_dims);
    return SUCCESS;
}

Result superpoint::UnInit() {
    Superpoint_Instant->DestroyResource();
}


Result superpoint::Preprocess(cv::Mat image) {
    struct  timeval tstart,tend, tmid;
    double timeuse;
    gettimeofday(&tstart,NULL);

    image.convertTo(image, CV_32F);
    cv::Mat normalized_image = image / 255.0;

    // int count = 0;
    // for (int i = 0; i < normalized_image.rows; ++i) {
    //     for (int j = 0; j < normalized_image.cols; ++j) {
    //         if (count >= 20) break;  // 限制打印 20 个元素
    //         std::cout << normalized_image.at<float>(i, j) << " ";
    //         ++count;
    //     }
    //     if (count >= 20) break;
    // }
    // std::cout << std::endl;

    int ret = 0;           
                    
    ret = Superpoint_Instant->SetProcessInputItem(static_cast<void*>(normalized_image.data), 0);
  

    gettimeofday(&tend,NULL);
    timeuse = 1000000*(tend.tv_sec - tstart.tv_sec) + \
				(tend.tv_usec - tstart.tv_usec);
    std::cout<<"Superpoint preprocess time:  "<< timeuse/1000<<"ms"<<std::endl;

    return SUCCESS;
}




Result superpoint::Inference() {
    
    struct  timeval tstart,tend;
    double timeuse;
    gettimeofday(&tstart,NULL);

    Result ret = Superpoint_Instant->inference();

    gettimeofday(&tend,NULL);
    timeuse = 1000000*(tend.tv_sec - tstart.tv_sec) + \
				(tend.tv_usec - tstart.tv_usec);
    std::cout<<"Superpoint Inference time:  "<< timeuse/1000<<"ms"<<std::endl;

    return SUCCESS;
}

std::vector<float> superpoint::Postprocess() {
    
    struct  timeval tstart,tend;
    double timeuse;
    gettimeofday(&tstart,NULL);

    keypoints_.clear();
    descriptors_.clear();

    float* output_scores = (float *)Superpoint_Instant->GetInferenceOutputItem(0);  
    float* output_descriptors = (float *)Superpoint_Instant->GetInferenceOutputItem(1);  

  
    std::vector<float> scores_vec(output_scores, output_scores + g_modelWidth_ * g_modelHeight_);

    find_high_score_index(scores_vec, keypoints_, g_modelHeight_, g_modelWidth_, 0.005);
    
    remove_borders(keypoints_, scores_vec, 4, g_modelHeight_, g_modelWidth_);

    top_k_keypoints(keypoints_, scores_vec, 2000);
    // for (int i = 0; i < 100; i++) {
    //     std::cout<<keypoints_[i][0]<<" "<<keypoints_[i][1]<<std::endl;
    // }
    // std::cout << std::endl; // 换行
    // for (int i = 0; i < 100; ++i) {
    //     std::cout << scores_vec[i] << " ";
    // }
    sample_descriptors(keypoints_, output_descriptors, descriptors_, 256, g_modelHeight_ / 8, g_modelWidth_ / 8, 8);
    
    printf("descriptors_.size = %d\n", descriptors_[0].size());

  
    // std::cout << std::endl; // 换行

    std::cout<<"特征点个数:  "<<keypoints_.size()<<std::endl;
    gettimeofday(&tend,NULL);
    timeuse = 1000000*(tend.tv_sec - tstart.tv_sec) + \
				(tend.tv_usec - tstart.tv_usec);
    std::cout<<"Superpoint Postprocess time:  "<< timeuse/1000<<"ms"<<std::endl;

    return scores_vec;
}



void superpoint::find_high_score_index(std::vector<float> &scores, std::vector<std::vector<int>> &keypoints,
                                       int h, int w, float threshold) {
    std::vector<float> new_scores;
    for (int i = 0; i < scores.size(); ++i) {
        if (scores[i] > threshold) {
            std::vector<int> location = {int(i / w), i % w};
            keypoints.emplace_back(location);
            new_scores.push_back(scores[i]);
        }
    }
    scores.swap(new_scores);
}


void superpoint::remove_borders(std::vector<std::vector<int>> &keypoints, std::vector<float> &scores, int border,
                                int height,
                                int width) {
    std::vector<std::vector<int>> keypoints_selected;
    std::vector<float> scores_selected;
    for (int i = 0; i < keypoints.size(); ++i) {
        bool flag_h = (keypoints[i][0] >= border) && (keypoints[i][0] < (height - border));
        bool flag_w = (keypoints[i][1] >= border) && (keypoints[i][1] < (width - border));
        if (flag_h && flag_w) {
            keypoints_selected.push_back(std::vector<int>{keypoints[i][1], keypoints[i][0]});
            scores_selected.push_back(scores[i]);
        }
    }
    keypoints.swap(keypoints_selected);
    scores.swap(scores_selected);
}

std::vector<size_t> superpoint::sort_indexes(std::vector<float> &data) {
    std::vector<size_t> indexes(data.size());
    iota(indexes.begin(), indexes.end(), 0);
    sort(indexes.begin(), indexes.end(), [&data](size_t i1, size_t i2) { return data[i1] > data[i2]; });
    return indexes;
}

void superpoint::top_k_keypoints(std::vector<std::vector<int>> &keypoints, std::vector<float> &scores, int k) {
    if (k < keypoints.size() && k != -1) {
        std::vector<std::vector<int>> keypoints_top_k;
        std::vector<float> scores_top_k;
        std::vector<size_t> indexes = sort_indexes(scores);
        for (int i = 0; i < k; ++i) {
            keypoints_top_k.push_back(keypoints[indexes[i]]);
            scores_top_k.push_back(scores[indexes[i]]);
        }
        keypoints.swap(keypoints_top_k);
        scores.swap(scores_top_k);
    }
}


void superpoint::sample_descriptors(std::vector<std::vector<int>> &keypoints, float *descriptors,
                                    std::vector<std::vector<double>> &dest_descriptors, int dim, int h, int w, int s) {
    std::vector<std::vector<double>> keypoints_norm;
    this->normalize_keypoints(keypoints, keypoints_norm, h, w, s);
    this->grid_sample(descriptors, keypoints_norm, dest_descriptors, dim, h, w);
    this->normalize_descriptors(dest_descriptors);
}

void superpoint::normalize_keypoints(const std::vector<std::vector<int>> &keypoints, std::vector<std::vector<double>> &keypoints_norm,
                    int h, int w, int s) {
    for (auto &keypoint : keypoints) {
        std::vector<double> kp = {keypoint[0] - s / 2 + 0.5, keypoint[1] - s / 2 + 0.5};
        kp[0] = kp[0] / (w * s - s / 2 - 0.5);
        kp[1] = kp[1] / (h * s - s / 2 - 0.5);
        kp[0] = kp[0] * 2 - 1;
        kp[1] = kp[1] * 2 - 1;
        keypoints_norm.push_back(kp);
    }
}


int clip(int val, int max) {
    if (val < 0) return 0;
    return std::min(val, max - 1);
}

void superpoint::grid_sample(const float *input, std::vector<std::vector<double>> &grid,
                 std::vector<std::vector<double>> &output, int dim, int h, int w) {
    // descriptors 1, 256, image_height/8, image_width/8
    // keypoints 1, 1, number, 2
    // out 1, 256, 1, number
    for (auto &g : grid) {
        double ix = ((g[0] + 1) / 2) * (w - 1);
        double iy = ((g[1] + 1) / 2) * (h - 1);

        int ix_nw = clip(std::floor(ix), w);
        int iy_nw = clip(std::floor(iy), h);

        int ix_ne = clip(ix_nw + 1, w);
        int iy_ne = clip(iy_nw, h);

        int ix_sw = clip(ix_nw, w);
        int iy_sw = clip(iy_nw + 1, h);

        int ix_se = clip(ix_nw + 1, w);
        int iy_se = clip(iy_nw + 1, h);

        double nw = (ix_se - ix) * (iy_se - iy);
        double ne = (ix - ix_sw) * (iy_sw - iy);
        double sw = (ix_ne - ix) * (iy - iy_ne);
        double se = (ix - ix_nw) * (iy - iy_nw);

        std::vector<double> descriptor;
        for (int i = 0; i < dim; ++i) {
            // 256x60x106 dhw
            // x * height * depth + y * depth + z
            float nw_val = input[i * h * w + iy_nw * w + ix_nw];
            float ne_val = input[i * h * w + iy_ne * w + ix_ne];
            float sw_val = input[i * h * w + iy_sw * w + ix_sw];
            float se_val = input[i * h * w + iy_se * w + ix_se];
            descriptor.push_back(nw_val * nw + ne_val * ne + sw_val * sw + se_val * se);
        }
        output.push_back(descriptor);
    }

}

template<typename Iter_T>
float vector_normalize(Iter_T first, Iter_T last) {
    return sqrt(inner_product(first, last, first, 0.0));
}

void superpoint::normalize_descriptors(std::vector<std::vector<double>> &dest_descriptors) {
    for (auto &descriptor : dest_descriptors) {
        double norm_inv = 1.0 / vector_normalize(descriptor.begin(), descriptor.end());
        std::transform(descriptor.begin(), descriptor.end(), descriptor.begin(),
                       std::bind1st(std::multiplies<double>(), norm_inv));
    }
    // for (int i = 0; i < 30; ++i) {
    //     std::cout << dest_descriptors[0][i] << " ";
    // }
    // std::cout << std::endl; // 换行
}


void superpoint::visualization(const std::string &image_name, const cv::Mat &image) {
    cv::Mat image_display;
    if(image.channels() == 1)
        cv::cvtColor(image, image_display, cv::COLOR_GRAY2BGR);
    else
        image_display = image.clone();
    for (auto &keypoint : keypoints_) {
        cv::circle(image_display, cv::Point(int(keypoint[0]), int(keypoint[1])), 1, cv::Scalar(0, 0, 255), -1, 16);
    }
    cv::imwrite(image_name + "_visualization.jpg", image_display);
}
