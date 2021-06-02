//
// Created by hzh on 2021/4/21.
//

#ifndef JI_SRC_INFERENCE_SPDDETECTOR_H_
#define JI_SRC_INFERENCE_SPDDETECTOR_H_
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <inference_engine.hpp>

using namespace InferenceEngine;
using namespace cv;
using std::vector;

struct result
{
    cv::Rect rect;
    int idx;
    double prob;
};

class Detector
{

public:
    void init();

public:
    void ForwardNetwork(const cv::Mat &in_mat, std::vector<result> &detections);

    static Detector &getInstance()
    {
        static Detector instance;
        return instance;
    }

    bool parse_yolov5(const Blob::Ptr &blob, int net_grid, float cof_threshold,
                      vector<Rect> &o_rect, vector<float> &o_rect_cof, vector<int> &idxs);
    double sigmoid(double x);
    vector<int> get_anchors(int net_grid);

    Detector()
    {
        // EVLOG(INFO) << "create Detector instance" << std::endl;
        init();
    }
    int LoadNetworkFromFile(const std::string &xml_path, const std::string &bin_path, std::string platform = "CPU");

private:
    //openvino
    Core m_ie;
    //存储初始化获得的可执行网络
    ExecutableNetwork _network;
    OutputsDataMap _outputinfo;
    std::string _input_name;
    double _cof_threshold;      //置信度阈值,计算方法是框置信度乘以物品种类置信度
    double _nms_area_threshold; //nms最小重叠面积阈值
};

#endif //JI_SRC_INFERENCE_SPDDETECTOR_H_
