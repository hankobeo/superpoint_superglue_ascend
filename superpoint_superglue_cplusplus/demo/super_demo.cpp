#include "superpoint.h"
#include "superglue.h"

using namespace cv;


int main()
{
    uint32_t g_modelWidth = 1280;
    uint32_t g_modelHeight = 1024;

    std::string imgname1 = "../images/1_IR.jpg";
    std::string imgname2 = "../images/1_VL.jpg";

    const char* g_modelPath_extract = "../model/superpoint_1280_1024.om";                  // 关键点提取模型
    const char* g_modelPath_matching = "../model/superglue_outdoor_end2end.om";     // 图像配准模型

    superpoint * extract_left = new superpoint(g_modelPath_extract, g_modelWidth, g_modelHeight);
    superpoint * extract_right = new superpoint(g_modelPath_extract, g_modelWidth, g_modelHeight);

    Result ret = extract_left->Init();
    ret = extract_right->Init();

    cv::Mat img_left = cv::imread(imgname1);

    cv::resize(img_left, img_left, cv::Size(g_modelWidth, g_modelHeight));

    cv::Mat gray_img_left;
    cv::cvtColor(img_left, gray_img_left, cv::COLOR_BGR2GRAY);

    // cv::Mat zero_image = cv::Mat::zeros(1024, 1280, CV_8UC1);

    ret = extract_left->Preprocess(gray_img_left);
    ret = extract_left->Inference();
    std::vector<float> scores_left = extract_left->Postprocess();
    extract_left->visualization(imgname1, img_left);


    cv::Mat img_right = cv::imread(imgname2);
    cv::resize(img_right, img_right, cv::Size(g_modelWidth, g_modelHeight));
    cv::Mat gray_img_right;
    cv::cvtColor(img_right, gray_img_right, cv::COLOR_BGR2GRAY);

    ret = extract_right->Preprocess(gray_img_right);
    ret = extract_right->Inference();
    std::vector<float> scores_right = extract_right->Postprocess();
    extract_right->visualization(imgname2, img_right);

    std::vector<cv::DMatch> superglue_matches;

    superglue * matching = new superglue(g_modelPath_matching, g_modelWidth, g_modelHeight);

    ret = matching->Init();
    ret = matching->Preprocess(extract_left->keypoints_, scores_left, extract_left->descriptors_, extract_right->keypoints_, scores_right, extract_right->descriptors_);
    ret = matching->Inference();
    ret = matching->Postprocess(extract_left->keypoints_, extract_right->keypoints_, img_left, img_right);
    // ret = matching->Postprocess(extract_left->keypoints_, extract_right->keypoints_);
    return 0; 
}
