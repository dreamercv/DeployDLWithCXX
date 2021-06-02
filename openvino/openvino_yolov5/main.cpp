#include <iostream>
#include "detector.hpp"
using namespace std;

int main(int argc, char const *argv[])
{
    Detector *detector = new Detector;
    detector->init();
    Mat src = imread("/home/openvino_yolov5/models/bus.jpg");
    Mat osrc = src.clone();
    resize(osrc, osrc, Size(640, 640));
    vector<result> detected_objects;
    auto start = chrono::high_resolution_clock::now();
    detector->ForwardNetwork(src, detected_objects);
    auto end = chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    cout << "use " << diff.count() << " s" << endl;
    for (int i = 0; i < detected_objects.size(); ++i)
    {
        int xmin = detected_objects[i].rect.x;
        int ymin = detected_objects[i].rect.y;
        int width = detected_objects[i].rect.width;
        int height = detected_objects[i].rect.height;
        Rect rect(xmin, ymin, width, height); //左上坐标（x,y）和矩形的长(x)宽(y)
        cv::rectangle(osrc, rect, Scalar(0, 0, 255), 1, LINE_8, 0);
    }
    cv::imwrite("/home/openvino_yolov5/models/bus11.jpg", osrc);
    // imshow("result",osrc);
    // waitKey(0);
    return 0;
}