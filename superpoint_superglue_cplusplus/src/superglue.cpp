#include "superglue.h"


#define superglue_keypoints_thr 1024

superglue::superglue(const char* modelPath, uint32_t modelWidth,
                           uint32_t modelHeight)
    :g_modelWidth_(modelWidth), g_modelHeight_(modelHeight)
{
    g_modelPath_ = modelPath;
    out_dims = (aclmdlIODims*)malloc(sizeof(aclmdlIODims));
    in_dims = (aclmdlIODims*)malloc(sizeof(aclmdlIODims));
}

superglue::~superglue()
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


Result superglue::Init()
{
    Superglue_Instant = new basic(g_modelPath_);

    Result ret = Superglue_Instant->Init_atlas();
    ret = Superglue_Instant->CreateInput();
    ret = Superglue_Instant->CreateOutput();

    ret = Superglue_Instant->GetInputShape(in_dims);
    ret = Superglue_Instant->GetOutputShape(out_dims);
    return SUCCESS;
}


Result superglue::UnInit()
{

}


float* superglue::Preprocess_keypoints(std::vector<std::vector<int>> &keypoints)
{
    // normalize_keypoints(keypoints, 1280, 1024);

    float* keypoints_input = new float[superglue_keypoints_thr * 2]; // 分配内存
    std::fill(keypoints_input, keypoints_input + superglue_keypoints_thr * 2, 0.0f); // 初始化为 0

    for (size_t i = 0; i < keypoints.size() && i < superglue_keypoints_thr; ++i) {
        keypoints_input[i * 2] = static_cast<float>(keypoints[i][0]); // x 坐标
        keypoints_input[i * 2 + 1] = static_cast<float>(keypoints[i][1]); // y 坐标
    }
    return keypoints_input;

}

float* superglue::Preprocess_scores(std::vector<float> scores)
{
    float* scores_input = new float[superglue_keypoints_thr];
    std::fill(scores_input, scores_input + superglue_keypoints_thr, 0.0f); // 初始化为 0

    for (size_t i = 0; i < scores.size() && i < superglue_keypoints_thr; ++i) {
        scores_input[i] = scores[i];
    }

    return scores_input;
}


float* superglue::Preprocess_descriptors(std::vector<std::vector<double>> descriptors)
{
   // 创建一个 512x256 的二维向量，初始化为 0
    std::vector<std::vector<float>> descriptors_input(superglue_keypoints_thr, std::vector<float>(256, 0.0f));
    // 将 descriptors 的值复制到 descriptors_input 中
    for (size_t i = 0; i < descriptors.size() && i < superglue_keypoints_thr; ++i) {
        for (size_t j = 0; j < 256; ++j) {
            descriptors_input[i][j] = static_cast<float>(descriptors[i][j]);
        }
    }

    float* descriptors_output = new float[superglue_keypoints_thr * 256];
    for (size_t j = 0; j < 256; ++j) {
        for (size_t i = 0; i < superglue_keypoints_thr; ++i) {
            descriptors_output[j * superglue_keypoints_thr + i] = descriptors_input[i][j];
        }
    }

    return descriptors_output;
}


Result superglue::Preprocess(std::vector<std::vector<int>> keypoints0, 
                             std::vector<float> scores0,
                             std::vector<std::vector<double>> descriptors0,
                             std::vector<std::vector<int>> keypoints1, 
                             std::vector<float> scores1,
                             std::vector<std::vector<double>> descriptors1)
{
    struct  timeval tstart,tend, tmid;
    double timeuse;
    gettimeofday(&tstart,NULL);       
    

    // printf("keypoints0纬度为:%dx%d, scores0纬度为:%d, descriptors0纬度为%dx%d\n", keypoints0.size(), keypoints0[0].size(), scores0.size(), descriptors0.size(), descriptors0[0].size());
    // printf("keypoints1纬度为:%dx%d, scores1纬度为:%d, descriptors1纬度为%dx%d\n", keypoints1.size(), keypoints1[0].size(), scores1.size(), descriptors1.size(), descriptors1[0].size());

    float * keypoints0_input = Preprocess_keypoints(keypoints0);
    float * scores0_input = Preprocess_scores(scores0);
    float * descriptors0_input = Preprocess_descriptors(descriptors0);
    float * keypoints1_input = Preprocess_keypoints(keypoints1);
    float * scores1_input = Preprocess_scores(scores1);
    float * descriptors1_input = Preprocess_descriptors(descriptors1);

    int ret = Superglue_Instant->SetProcessInputItem(static_cast<void*>(keypoints0_input), 0);
    ret = Superglue_Instant->SetProcessInputItem(static_cast<void*>(scores0_input), 1);
    ret = Superglue_Instant->SetProcessInputItem(static_cast<void*>(descriptors0_input), 2);
    ret = Superglue_Instant->SetProcessInputItem(static_cast<void*>(keypoints1_input), 3);
    ret = Superglue_Instant->SetProcessInputItem(static_cast<void*>(scores1_input), 4);
    ret = Superglue_Instant->SetProcessInputItem(static_cast<void*>(descriptors1_input), 5);

  

    gettimeofday(&tend,NULL);
    timeuse = 1000000*(tend.tv_sec - tstart.tv_sec) + \
				(tend.tv_usec - tstart.tv_usec);
    std::cout<<"Superglue preprocess time:  "<< timeuse/1000<<"ms"<<std::endl;

    return SUCCESS;
}

Result superglue::Inference()
{
    struct  timeval tstart,tend;
    double timeuse;
    gettimeofday(&tstart,NULL);

    Result ret = Superglue_Instant->inference();

    gettimeofday(&tend,NULL);
    timeuse = 1000000*(tend.tv_sec - tstart.tv_sec) + \
				(tend.tv_usec - tstart.tv_usec);
    std::cout<<"Superglue Inference time:  "<< timeuse/1000<<"ms"<<std::endl;

    return SUCCESS;
}


void where_negative_one(const int *flag_data, const int *data, int size, std::vector<int> &indices) {
    for (int i = 0; i < size; ++i) {
        if (flag_data[i] == 1) {
            indices.push_back(data[i]);
        } else {
            indices.push_back(-1);
        }
    }
}

void max_matrix(const float *data, int *indices, float *values, int h, int w, int dim) {
    if (dim == 2) {
        for (int i = 0; i < h - 1; ++i) {
            float max_value = -FLT_MAX;
            int max_indices = 0;
            for (int j = 0; j < w - 1; ++j) {
                if (max_value < data[i * w + j]) {
                    max_value = data[i * w + j];
                    max_indices = j;
                }
            }
            values[i] = max_value;
            indices[i] = max_indices;
        }
    } else if (dim == 1) {
        for (int i = 0; i < w - 1; ++i) {
            float max_value = -FLT_MAX;
            int max_indices = 0;
            for (int j = 0; j < h - 1; ++j) {
                if (max_value < data[j * w + i]) {
                    max_value = data[j * w + i];
                    max_indices = j;
                }
            }
            values[i] = max_value;
            indices[i] = max_indices;
        }
    }
}

void equal_gather(const int *indices0, const int *indices1, int *mutual, int size) {
    for (int i = 0; i < size; ++i) {
        if (indices0[indices1[i]] == i) {
            mutual[i] = 1;
        } else {
            mutual[i] = 0;
        }
    }
}

void where_exp(const int *flag_data, float *data, std::vector<double> &mscores0, int size) {
    for (int i = 0; i < size; ++i) {
        if (flag_data[i] == 1) {
            mscores0.push_back(std::exp(data[i]));
        } else {
            mscores0.push_back(0);
        }
    }
}

void where_gather(const int *flag_data, int *indices, std::vector<double> &mscores0, std::vector<double> &mscores1,
                  int size) {
    for (int i = 0; i < size; ++i) {
        if (flag_data[i] == 1) {
            mscores1.push_back(mscores0[indices[i]]);
        } else {
            mscores1.push_back(0);
        }
    }
}

void and_threshold(const int *mutual0, int *valid0, const std::vector<double> &mscores0, double threhold) {
    for (int i = 0; i < mscores0.size(); ++i) {
        if (mutual0[i] == 1 && mscores0[i] > threhold) {
            valid0[i] = 1;
        } else {
            valid0[i] = 0;
        }
    }
}

void and_gather(const int *mutual1, const int *valid0, const int *indices1, int *valid1, int size) {
    for (int i = 0; i < size; ++i) {
        if (mutual1[i] == 1 && valid0[indices1[i]] == 1) {
            valid1[i] = 1;
        } else {
            valid1[i] = 0;
        }
    }
}

void decode(float *scores, int h, int w, std::vector<int> &indices0, std::vector<int> &indices1,
            std::vector<double> &mscores0, std::vector<double> &mscores1) {
    auto *max_indices0 = new int[h - 1];
    auto *max_indices1 = new int[w - 1];
    auto *max_values0 = new float[h - 1];
    auto *max_values1 = new float[w - 1];
    max_matrix(scores, max_indices0, max_values0, h, w, 2);
    max_matrix(scores, max_indices1, max_values1, h, w, 1);
    auto *mutual0 = new int[h - 1];
    auto *mutual1 = new int[w - 1];
    equal_gather(max_indices1, max_indices0, mutual0, h - 1);
    equal_gather(max_indices0, max_indices1, mutual1, w - 1);
    where_exp(mutual0, max_values0, mscores0, h - 1);
    where_gather(mutual1, max_indices1, mscores0, mscores1, w - 1);
    auto *valid0 = new int[h - 1];
    auto *valid1 = new int[w - 1];
    and_threshold(mutual0, valid0, mscores0, 0.2);
    and_gather(mutual1, valid0, max_indices1, valid1, w - 1);
    where_negative_one(valid0, max_indices0, h - 1, indices0);
    where_negative_one(valid1, max_indices1, w - 1, indices1);
    delete[] max_indices0;
    delete[] max_indices1;
    delete[] max_values0;
    delete[] max_values1;
    delete[] mutual0;
    delete[] mutual1;
    delete[] valid0;
    delete[] valid1;
}

Result superglue::Postprocess(std::vector<std::vector<int>> keypoints0, std::vector<std::vector<int>> keypoints1)
{
    struct  timeval tstart,tend;
    double timeuse;
    gettimeofday(&tstart,NULL);


    float* output_scores = (float *)Superglue_Instant->GetInferenceOutputItem(0);  
    for (int i = 0; i < 30; ++i) {
        std::cout << output_scores[i] << " ";
    }
    std::cout << std::endl; // 换行

    printf("%d, %d, %d\n", out_dims->dims[0], out_dims->dims[1], out_dims->dims[2]); // 1xhxw

    std::vector<int> indices0;
    std::vector<int> indices1;
    std::vector<double> mscores0;
    std::vector<double> mscores1;
    decode(output_scores, out_dims->dims[1], out_dims->dims[2], indices0, indices1, mscores0, mscores1);

    std::vector<cv::DMatch> matches;
    int num_match = 0;
    std::vector<cv::Point2f> points0, points1;
 
    for(size_t i = 0; i < indices0.size(); i++){
        if (indices0[i] >= 0 && indices0[i] < indices1.size() && indices1[indices0[i]] == i)
        {
            double d = 1.0 - (mscores0[i] + mscores1[indices0[i]]) / 2.0;
            matches.emplace_back(i, indices0[i], d);
            points0.emplace_back(static_cast<float>(keypoints0[i][1]), static_cast<float>(keypoints0[i][2]));
            points1.emplace_back(static_cast<float>(keypoints1[indices0[i]][1]), static_cast<float>(keypoints1[indices0[i]][2]));
            num_match++;
        }
    }


    
    std::cout<<"Superglue匹配个数:  "<< num_match <<std::endl;


    gettimeofday(&tend,NULL);
    timeuse = 1000000*(tend.tv_sec - tstart.tv_sec) + \
				(tend.tv_usec - tstart.tv_usec);
    std::cout<<"Superglue Postprocess time:  "<< timeuse/1000<<"ms"<<std::endl;


}

// 在图像上绘制关键点
void drawKeypointsOnImage(const std::vector<cv::KeyPoint>& points, cv::Mat& img, const cv::Scalar& color = cv::Scalar(0, 255, 0), int radius = 2)
{
    // 遍历所有关键点
    for (const auto& point : points) {
        // std::cout << "KeyPoint Coordinates: (" << point.pt.x << ", " << point.pt.y << ")" << std::endl;
        // 绘制圆圈表示关键点
        cv::circle(img, point.pt, radius, color, -1); // 填充圆
    }
    cv::imwrite("./img1.jpg", img);
}


void drawMatchesWithRandomColors(const std::vector<cv::KeyPoint>& points0, 
                                 const std::vector<cv::KeyPoint>& points1, 
                                 cv::Mat& img0, 
                                 cv::Mat& img1, 
                                 int match_nums,
                                 const std::string& output_file = "./matches.png"
                                 ) 
{
    // 创建并排显示的图像
    cv::Mat match_image(std::max(img0.rows, img1.rows), img0.cols + img1.cols, img0.type());
    img0.copyTo(match_image(cv::Rect(0, 0, img0.cols, img0.rows)));
    img1.copyTo(match_image(cv::Rect(img0.cols, 0, img1.cols, img1.rows)));

    // 随机数生成器
    cv::RNG rng;
    
    // 确保 points0 和 points1 长度一致
    if (points0.size() != points1.size()) {
        throw std::invalid_argument("points0 and points1 must have the same size.");
    }

    // 添加 match_nums 文本到左上角
    std::string text = "Matches: " + std::to_string(match_nums);
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 1.5;    // 增大字体比例（原来是0.7）
    int thickness = 3;         // 增加线条厚度（原来是2）
    cv::Scalar textColor(255, 255, 255); // 白色文字
    cv::Point textOrg(15, 50); // 调整位置以适应更大文字 (原来是10,30)

    // 绘制文字
    cv::putText(match_image, text, textOrg, fontFace, fontScale, textColor, thickness);


    // 遍历所有关键点
    for (size_t i = 0; i < points0.size(); ++i) {
        // 获取对应的关键点坐标
        cv::Point2f pt0 = points0[i].pt;
        cv::Point2f pt1 = points1[i].pt;
        pt1.x += img0.cols; // 偏移到拼接图像的右侧

        // 随机生成颜色
        cv::Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

        // 绘制连线和关键点
        cv::line(match_image, pt0, pt1, color, 1);
        cv::circle(match_image, pt0, 3, color, -1);
        cv::circle(match_image, pt1, 3, color, -1);
    }

    // 保存结果
    cv::imwrite(output_file, match_image);
}


Result superglue::Postprocess(std::vector<std::vector<int>> keypoints0, std::vector<std::vector<int>> keypoints1, cv::Mat img1, cv::Mat img2)
{
    struct  timeval tstart,tend;
    double timeuse;
    gettimeofday(&tstart,NULL);


    int64_t* matches0 = (int64_t *)Superglue_Instant->GetInferenceOutputItem(0);  
    // for (int i = 0; i < 30; ++i) {
    //     std::cout << matches0[i] << " ";
    // }
    // std::cout << std::endl; // 换行

    int64_t* matches1 = (int64_t *)Superglue_Instant->GetInferenceOutputItem(1); 
    // for (int i = 0; i < 30; ++i) {
    //     std::cout << matches1[i] << " ";
    // }
    // std::cout << std::endl; // 换行

    float* matches_scores0 = (float *)Superglue_Instant->GetInferenceOutputItem(2); 
    // for (int i = 0; i < 300; ++i) {
    //     std::cout << matches_scores0[i] << " ";
    // }
    // std::cout << std::endl; // 换行
    
    float* matches_scores1 = (float *)Superglue_Instant->GetInferenceOutputItem(3); 

    std::vector<cv::DMatch> matches;
    std::vector<cv::KeyPoint> points0, points1;

    int index = 0;
    
    for (size_t i = 0; i < keypoints0.size(); ++i) {
        if (matches0[i] >= 0 && matches0[i] < static_cast<int64_t>(keypoints1.size()) && matches1[matches0[i]] == static_cast<int64_t>(i)) {
            float d = 1.0f - (matches_scores0[i] + matches_scores1[matches0[i]]) / 2.0f;
            // printf("i = %d, matches0[i] = %d, matches1[matches0[i]] = %d\n", i, matches0[i], matches1[matches0[i]]);

            // std::cout << "i = " << i << "时, 左图坐标为(" << keypoints0[i][0] << ", " << keypoints0[i][1] << ")\n";
            // std::cout << "i = " << i << "时, 右图坐标为(" << keypoints1[static_cast<int>(matches0[i])][0] << ", " << keypoints1[static_cast<int>(matches0[i])][1] << ")\n";

            // 构建匹配关系
            matches.emplace_back(i, matches0[i], d);

            // 存储关键点
            points0.emplace_back(keypoints0[i][0], keypoints0[i][1], 5.0f);
            points1.emplace_back(keypoints1[matches0[i]][0], keypoints1[matches0[i]][1], 5.0f);

            index++;
        }
    }

    printf("match nums = %d\n", index);
    std::cout<<"valid_keypoints_left size = "<<points0.size()<<std::endl;
    std::cout<<"valid_keypoints_right size = "<<points1.size()<<std::endl;

    drawMatchesWithRandomColors(points0, points1, img1, img2, index);

    // drawKeypointsOnImage(points1, img2);

    gettimeofday(&tend,NULL);
    timeuse = 1000000*(tend.tv_sec - tstart.tv_sec) + \
				(tend.tv_usec - tstart.tv_usec);
    std::cout<<"Superglue Postprocess time:  "<< timeuse/1000<<"ms"<<std::endl;


}

// void superglue::normalize_keypoints(std::vector<std::vector<int>> &keypoints, int width, int height) {
//     float max_dim = std::max(width, height); // 获取宽度和高度中的最大值
//     float scale_factor = max_dim * 0.7f;  // 缩放因子，0.7 是一个经验值，可以根据需要调整

//     // 对每个特征点进行归一化
//     for (auto& point : keypoints) {
//         // 假设 point[0] 是 x 坐标，point[1] 是 y 坐标
//         point[0] = static_cast<int>((point[0] - width / 2.0f) / scale_factor); // 归一化 x 坐标
//         point[1] = static_cast<int>((point[1] - height / 2.0f) / scale_factor); // 归一化 y 坐标
//     }
// }

void superglue::normalize_keypoints(std::vector<std::vector<float>>& keypoints, int width, int height) {
    // Use float instead of int for keypoints to maintain precision
    float w = static_cast<float>(width);
    float h = static_cast<float>(height);
    float max_dim = std::max(w, h);
    float scale_factor = max_dim * 0.7f;
    
    // Center coordinates
    float center_x = w / 2.0f;
    float center_y = h / 2.0f;

    // Normalize each keypoint
    for (auto& point : keypoints) {
        // Assuming point[0] is x and point[1] is y
        point[0] = (point[0] - center_x) / scale_factor;
        point[1] = (point[1] - center_y) / scale_factor;
    }
}