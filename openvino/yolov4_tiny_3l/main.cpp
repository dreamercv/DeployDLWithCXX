#include <iostream>
#include <random>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

#include <inference_engine.hpp>
#include <ngraph/ngraph.hpp>

using namespace InferenceEngine;
using namespace cv;
using std::vector;
using namespace std;

template <typename T>
void matToBlob(const cv::Mat &frame, InferenceEngine::Blob::Ptr &frameBlob, int batchIndex = 0)
{
    InferenceEngine::SizeVector blobSize = frameBlob->getTensorDesc().getDims();
    const size_t width = blobSize[3];
    const size_t height = blobSize[2];
    const size_t channels = blobSize[1];
    T *blob_data = frameBlob->buffer().as<T *>();

    cv::Mat resized_image(frame);
    if (static_cast<int>(width) != frame.size().width ||
        static_cast<int>(height) != frame.size().height)
    {
        cv::resize(frame, resized_image, cv::Size(width, height));
    }

    int batchOffset = batchIndex * width * height * channels;

    for (size_t c = 0; c < channels; c++)
    {
        for (size_t h = 0; h < height; h++)
        {
            for (size_t w = 0; w < width; w++)
            {
                blob_data[batchOffset + c * width * height + h * width + w] =
                    resized_image.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
}

void FrameToBlob(const cv::Mat &frame, InferRequest::Ptr &inferRequest, const std::string &inputName)
{

    /* Resize and copy data from the image to the input blob */
    Blob::Ptr frameBlob = inferRequest->GetBlob(inputName);
    matToBlob<uint8_t>(frame, frameBlob);
}

static int EntryIndex(int side, int lcoords, int lclasses, int location, int entry)
{
    int n = location / (side * side);
    int loc = location % (side * side);
    return n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc;
}

struct DetectionObject
{
    int xmin, ymin, xmax, ymax, class_id;
    float confidence;

    DetectionObject(double x, double y, double h, double w, int class_id, float confidence, float h_scale, float w_scale)
    {
        this->xmin = static_cast<int>((x - w / 2) * w_scale);
        this->ymin = static_cast<int>((y - h / 2) * h_scale);
        this->xmax = static_cast<int>(this->xmin + w * w_scale);
        this->ymax = static_cast<int>(this->ymin + h * h_scale);
        this->class_id = class_id;
        this->confidence = confidence;
    }

    bool operator<(const DetectionObject &s2) const
    {
        return this->confidence < s2.confidence;
    }
    bool operator>(const DetectionObject &s2) const
    {
        return this->confidence > s2.confidence;
    }
};

double IOU(const DetectionObject &box_1, const DetectionObject &box_2)
{
    double width_of_overlap_area = fmin(box_1.xmax, box_2.xmax) - fmax(box_1.xmin, box_2.xmin);
    double height_of_overlap_area = fmin(box_1.ymax, box_2.ymax) - fmax(box_1.ymin, box_2.ymin);
    double area_of_overlap;
    if (width_of_overlap_area < 0 || height_of_overlap_area < 0)
        area_of_overlap = 0;
    else
        area_of_overlap = width_of_overlap_area * height_of_overlap_area;
    double box_1_area = (box_1.ymax - box_1.ymin) * (box_1.xmax - box_1.xmin);
    double box_2_area = (box_2.ymax - box_2.ymin) * (box_2.xmax - box_2.xmin);
    double area_of_union = box_1_area + box_2_area - area_of_overlap;
    return area_of_overlap / area_of_union;
}

class YoloParams
{
    template <typename T>
    void computeAnchors(const std::vector<T> &mask)
    {
        std::vector<float> maskedAnchors(num * 2);
        for (int i = 0; i < num; ++i)
        {
            maskedAnchors[i * 2] = anchors[mask[i] * 2];
            maskedAnchors[i * 2 + 1] = anchors[mask[i] * 2 + 1];
        }
        anchors = maskedAnchors;
    }

public:
    int num = 0, classes = 0, coords = 0;
    std::vector<float> anchors = {10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0,
                                  156.0, 198.0, 373.0, 326.0};

    YoloParams() {}

    YoloParams(const std::shared_ptr<ngraph::op::RegionYolo> regionYolo)
    {
        coords = regionYolo->get_num_coords();
        classes = regionYolo->get_num_classes();
        anchors = regionYolo->get_anchors();
        auto mask = regionYolo->get_mask();
        num = mask.size();

        computeAnchors(mask);
    }

    YoloParams(CNNLayer::Ptr layer)
    {
        if (layer->type != "RegionYolo")
            throw std::runtime_error("Invalid output type: " + layer->type + ". RegionYolo expected");

        num = layer->GetParamAsInt("num");
        coords = layer->GetParamAsInt("coords");
        classes = layer->GetParamAsInt("classes");

        try
        {
            anchors = layer->GetParamAsFloats("anchors");
        }
        catch (...)
        {
        }
        try
        {
            auto mask = layer->GetParamAsInts("mask");
            num = mask.size();

            computeAnchors(mask);
        }
        catch (...)
        {
        }
    }
};

void ParseYOLOV3Output(const CNNNetwork &cnnNetwork, const std::string &output_name,
                       const Blob::Ptr &blob, const unsigned long resized_im_h,
                       const unsigned long resized_im_w, const unsigned long original_im_h,
                       const unsigned long original_im_w,
                       const double threshold, std::vector<DetectionObject> &objects)
{

    const int out_blob_h = static_cast<int>(blob->getTensorDesc().getDims()[2]);
    const int out_blob_w = static_cast<int>(blob->getTensorDesc().getDims()[3]);

    // --------------------------- Extracting layer parameters -------------------------------------
    YoloParams params;

    auto ngraphFunction = cnnNetwork.getFunction();

    for (const auto op : ngraphFunction->get_ops())
    {
        if (op->get_friendly_name() == output_name)
        {
            auto regionYolo = std::dynamic_pointer_cast<ngraph::op::RegionYolo>(op);
            params = regionYolo;
            break;
        }
    }

    auto side = out_blob_h;
    auto side_square = side * side;
    const float *output_blob = blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
    // --------------------------- Parsing YOLO Region output -------------------------------------
    for (int i = 0; i < side_square; ++i)
    {
        int row = i / side;
        int col = i % side;
        for (int n = 0; n < params.num; ++n)
        {
            int obj_index = EntryIndex(side, params.coords, params.classes, n * side * side + i, params.coords);
            int box_index = EntryIndex(side, params.coords, params.classes, n * side * side + i, 0);
            float scale = output_blob[obj_index];
            if (scale < threshold)
                continue;
            double x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w;
            double y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h;
            double height = std::exp(output_blob[box_index + 3 * side_square]) * params.anchors[2 * n + 1];
            double width = std::exp(output_blob[box_index + 2 * side_square]) * params.anchors[2 * n];
            for (int j = 0; j < params.classes; ++j)
            {
                int class_index = EntryIndex(side, params.coords, params.classes, n * side_square + i, params.coords + 1 + j);
                float prob = scale * output_blob[class_index];
                if (prob < threshold)
                    continue;
                DetectionObject obj(x, y, height, width, j, prob,
                                    static_cast<float>(original_im_h) / static_cast<float>(resized_im_h),
                                    static_cast<float>(original_im_w) / static_cast<float>(resized_im_w));
                objects.push_back(obj);
            }
        }
    }
}

int main(int argc, char *argv[])
{

    /** This demo covers a certain topology and cannot be generalized for any object detection **/
    std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

    // read input (video) frame
    std::string xml_path = "/home/yolov4_tiny_3l/models/yolov4-tiny-3l_best.xml";
    std::string bin_path = "";
    std::string image_path = "/home/yolov4_tiny_3l/models/demo.jpg";

    cv::Mat frame = cv::imread(image_path);

    cv::Mat next_frame = cv::imread(image_path);
    size_t height = frame.size().height;
    size_t width = frame.size().width;

    Core ie;
    // --------------- 2. Reading the IR generated by the Model Optimizer (.xml and .bin files) ------------
    auto cnnNetwork = ie.ReadNetwork(xml_path);

    /** YOLOV3-based network should have one input and three output **/
    // --------------------------- 3. Configuring input and output -----------------------------------------
    // --------------------------------- Preparing input blobs ---------------------------------------------
    InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
    InputInfo::Ptr &input = inputInfo.begin()->second;
    auto inputName = inputInfo.begin()->first;
    input->setPrecision(Precision::U8); //U8

    input->getInputData()->setLayout(Layout::NCHW);

    ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
    SizeVector &inSizeVector = inputShapes.begin()->second;
    inSizeVector[0] = 1; // set batch to 1
    cnnNetwork.reshape(inputShapes);
    // --------------------------------- Preparing output blobs -------------------------------------------

    OutputsDataMap outputInfo(cnnNetwork.getOutputsInfo());
    for (auto &output : outputInfo)
    {
        output.second->setPrecision(Precision::FP32);
        output.second->setLayout(Layout::NCHW);
    }

    // --------------------------- 4. Loading model to the device ------------------------------------------

    ExecutableNetwork network = ie.LoadNetwork(cnnNetwork, "CPU");

    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 5. Creating infer request -----------------------------------------------
    InferRequest::Ptr infer_request = network.CreateInferRequestPtr();
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 6. Doing inference -----------------------------------------------------

    FrameToBlob(frame, infer_request, inputName);

    infer_request->Infer();

    // ---------------------------Processing output blobs--------------------------------------------------
    // Processing results of the CURRENT request

    const TensorDesc &inputDesc = inputInfo.begin()->second.get()->getTensorDesc();
    unsigned long resized_im_h = 288; //getTensorHeight(inputDesc);
    unsigned long resized_im_w = 288; //getTensorWidth(inputDesc);
    std::vector<DetectionObject> objects;
    // Parsing outputs
    for (auto &output : outputInfo)
    {
        auto output_name = output.first;

        std::string sub_str = "Split";
        string::size_type idx = output_name.find(sub_str);
        if (idx != string::npos)
        {
            continue;
        }

        std::cout << output_name << std::endl;
        Blob::Ptr blob = infer_request->GetBlob(output_name);

        ParseYOLOV3Output(cnnNetwork, output_name, blob, resized_im_h, resized_im_w, height, width, 0.1, objects);
    }

    // Filtering overlapping boxes
    std::sort(objects.begin(), objects.end(), std::greater<DetectionObject>());
    for (size_t i = 0; i < objects.size(); ++i)
    {
        if (objects[i].confidence == 0)
            continue;
        for (size_t j = i + 1; j < objects.size(); ++j)
            if (IOU(objects[i], objects[j]) >= 0.4)
                objects[j].confidence = 0;
    }

    std::cout << objects.size() << std::endl;
    // Drawing boxes
    for (auto &object : objects)
    {
        if (object.confidence < 0.3)
            continue;
        auto label = object.class_id;
        float confidence = object.confidence;
        if (confidence >= 0.3)
        {
            std::string ss;
            ss = std::to_string(label) + ":" + std::to_string(confidence);
            cv::Point origin;
            origin.x = object.xmin;
            origin.y = object.ymin;
            int font_face = cv::FONT_HERSHEY_COMPLEX;
            double font_scale = 1;
            int thickness = 1;
            cv::putText(frame, ss, origin, font_face, font_scale, cv::Scalar(0, 255, 0), thickness);

            /** Drawing only objects when >confidence_threshold probability **/
            cv::rectangle(frame, cv::Point2f(static_cast<float>(object.xmin), static_cast<float>(object.ymin)),
                          cv::Point2f(static_cast<float>(object.xmax), static_cast<float>(object.ymax)), cv::Scalar(0, 0, 255));
        }
    }
    cv::imwrite("/home/yolov4_tiny_3l/models/demo3.jpg", frame);

    return 0;
}

//================================异步===============================//

// #include <gflags/gflags.h>
// #include <functional>
// #include <iostream>
// #include <fstream>
// #include <random>
// #include <memory>
// #include <chrono>
// #include <vector>
// #include <string>
// #include <algorithm>
// #include <iterator>

// #include <opencv2/opencv.hpp>
// #include <opencv2/dnn/dnn.hpp>

// #include <inference_engine.hpp>
// #include <ngraph/ngraph.hpp>

// using namespace InferenceEngine;
// using namespace cv;
// using std::vector;

// bool FLAGS_auto_resize = true;

// template <typename T>
// void matU8ToBlob(const cv::Mat &orig_image, InferenceEngine::Blob::Ptr &blob, int batchIndex = 0)
// {
//     InferenceEngine::SizeVector blobSize = blob->getTensorDesc().getDims();
//     const size_t width = blobSize[3];
//     const size_t height = blobSize[2];
//     const size_t channels = blobSize[1];
//     if (static_cast<size_t>(orig_image.channels()) != channels)
//     {
//         THROW_IE_EXCEPTION << "The number of channels for net input and image must match";
//     }
//     T *blob_data = blob->buffer().as<T *>();

//     cv::Mat resized_image(orig_image);
//     if (static_cast<int>(width) != orig_image.size().width ||
//         static_cast<int>(height) != orig_image.size().height)
//     {
//         cv::resize(orig_image, resized_image, cv::Size(width, height));
//     }

//     int batchOffset = batchIndex * width * height * channels;

//     if (channels == 1)
//     {
//         for (size_t h = 0; h < height; h++)
//         {
//             for (size_t w = 0; w < width; w++)
//             {
//                 blob_data[batchOffset + h * width + w] = resized_image.at<uchar>(h, w);
//             }
//         }
//     }
//     else if (channels == 3)
//     {
//         for (size_t c = 0; c < channels; c++)
//         {
//             for (size_t h = 0; h < height; h++)
//             {
//                 for (size_t w = 0; w < width; w++)
//                 {
//                     blob_data[batchOffset + c * width * height + h * width + w] =
//                         resized_image.at<cv::Vec3b>(h, w)[c];
//                 }
//             }
//         }
//     }
//     else
//     {
//         THROW_IE_EXCEPTION << "Unsupported number of channels";
//     }
// }

// InferenceEngine::Blob::Ptr wrapMat2Blob(const cv::Mat &mat)
// {
//     size_t channels = mat.channels();
//     size_t height = mat.size().height;
//     size_t width = mat.size().width;

//     size_t strideH = mat.step.buf[0];
//     size_t strideW = mat.step.buf[1];

//     bool is_dense =
//         strideW == channels &&
//         strideH == channels * width;

//     if (!is_dense)
//         THROW_IE_EXCEPTION
//             << "Doesn't support conversion from not dense cv::Mat";

//     InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::U8,
//                                       {1, channels, height, width},
//                                       InferenceEngine::Layout::NHWC);

//     return InferenceEngine::make_shared_blob<uint8_t>(tDesc, mat.data);
// }

// void FrameToBlob(const cv::Mat &frame, InferRequest::Ptr &inferRequest, const std::string &inputName)
// {
//     if (FLAGS_auto_resize)
//     {
//         /* Just set input blob containing read image. Resize and layout conversion will be done automatically */
//         inferRequest->SetBlob(inputName, wrapMat2Blob(frame));
//     }
//     else
//     {
//         /* Resize and copy data from the image to the input blob */
//         Blob::Ptr frameBlob = inferRequest->GetBlob(inputName);
//         matU8ToBlob<uint8_t>(frame, frameBlob);
//     }
// }

// static int EntryIndex(int side, int lcoords, int lclasses, int location, int entry)
// {
//     int n = location / (side * side);
//     int loc = location % (side * side);
//     return n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc;
// }

// struct DetectionObject
// {
//     int xmin, ymin, xmax, ymax, class_id;
//     float confidence;

//     DetectionObject(double x, double y, double h, double w, int class_id, float confidence, float h_scale, float w_scale)
//     {
//         this->xmin = static_cast<int>((x - w / 2) * w_scale);
//         this->ymin = static_cast<int>((y - h / 2) * h_scale);
//         this->xmax = static_cast<int>(this->xmin + w * w_scale);
//         this->ymax = static_cast<int>(this->ymin + h * h_scale);
//         this->class_id = class_id;
//         this->confidence = confidence;
//     }

//     bool operator<(const DetectionObject &s2) const
//     {
//         return this->confidence < s2.confidence;
//     }
//     bool operator>(const DetectionObject &s2) const
//     {
//         return this->confidence > s2.confidence;
//     }
// };

// double IntersectionOverUnion(const DetectionObject &box_1, const DetectionObject &box_2)
// {
//     double width_of_overlap_area = fmin(box_1.xmax, box_2.xmax) - fmax(box_1.xmin, box_2.xmin);
//     double height_of_overlap_area = fmin(box_1.ymax, box_2.ymax) - fmax(box_1.ymin, box_2.ymin);
//     double area_of_overlap;
//     if (width_of_overlap_area < 0 || height_of_overlap_area < 0)
//         area_of_overlap = 0;
//     else
//         area_of_overlap = width_of_overlap_area * height_of_overlap_area;
//     double box_1_area = (box_1.ymax - box_1.ymin) * (box_1.xmax - box_1.xmin);
//     double box_2_area = (box_2.ymax - box_2.ymin) * (box_2.xmax - box_2.xmin);
//     double area_of_union = box_1_area + box_2_area - area_of_overlap;
//     return area_of_overlap / area_of_union;
// }

// class YoloParams
// {
//     template <typename T>
//     void computeAnchors(const std::vector<T> &mask)
//     {
//         std::vector<float> maskedAnchors(num * 2);
//         for (int i = 0; i < num; ++i)
//         {
//             maskedAnchors[i * 2] = anchors[mask[i] * 2];
//             maskedAnchors[i * 2 + 1] = anchors[mask[i] * 2 + 1];
//         }
//         anchors = maskedAnchors;
//     }

// public:
//     int num = 0, classes = 0, coords = 0;
//     std::vector<float> anchors = {10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0,
//                                   156.0, 198.0, 373.0, 326.0};

//     YoloParams() {}

//     YoloParams(const std::shared_ptr<ngraph::op::RegionYolo> regionYolo)
//     {
//         coords = regionYolo->get_num_coords();
//         classes = regionYolo->get_num_classes();
//         anchors = regionYolo->get_anchors();
//         auto mask = regionYolo->get_mask();
//         num = mask.size();

//         computeAnchors(mask);
//     }

//     YoloParams(CNNLayer::Ptr layer)
//     {
//         if (layer->type != "RegionYolo")
//             throw std::runtime_error("Invalid output type: " + layer->type + ". RegionYolo expected");

//         num = layer->GetParamAsInt("num");
//         coords = layer->GetParamAsInt("coords");
//         classes = layer->GetParamAsInt("classes");

//         try
//         {
//             anchors = layer->GetParamAsFloats("anchors");
//         }
//         catch (...)
//         {
//         }
//         try
//         {
//             auto mask = layer->GetParamAsInts("mask");
//             num = mask.size();

//             computeAnchors(mask);
//         }
//         catch (...)
//         {
//         }
//     }
// };

// void ParseYOLOV3Output(const CNNNetwork &cnnNetwork, const std::string &output_name,
//                        const Blob::Ptr &blob, const unsigned long resized_im_h,
//                        const unsigned long resized_im_w, const unsigned long original_im_h,
//                        const unsigned long original_im_w,
//                        const double threshold, std::vector<DetectionObject> &objects)
// {

//     const int out_blob_h = static_cast<int>(blob->getTensorDesc().getDims()[2]);
//     const int out_blob_w = static_cast<int>(blob->getTensorDesc().getDims()[3]);
//     if (out_blob_h != out_blob_w)
//         throw std::runtime_error("Invalid size of output " + output_name +
//                                  " It should be in NCHW layout and H should be equal to W. Current H = " + std::to_string(out_blob_h) +
//                                  ", current W = " + std::to_string(out_blob_h));

//     // --------------------------- Extracting layer parameters -------------------------------------
//     YoloParams params;
//     if (auto ngraphFunction = cnnNetwork.getFunction())
//     {
//         for (const auto op : ngraphFunction->get_ops())
//         {
//             if (op->get_friendly_name() == output_name)
//             {
//                 auto regionYolo = std::dynamic_pointer_cast<ngraph::op::RegionYolo>(op);
//                 if (!regionYolo)
//                 {
//                     throw std::runtime_error("Invalid output type: " +
//                                              std::string(regionYolo->get_type_info().name) + ". RegionYolo expected");
//                 }

//                 params = regionYolo;
//                 break;
//             }
//         }
//     }
//     else
//     {
//         throw std::runtime_error("Can't get ngraph::Function. Make sure the provided model is in IR version 10 or greater.");
//     }

//     auto side = out_blob_h;
//     auto side_square = side * side;
//     const float *output_blob = blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
//     // --------------------------- Parsing YOLO Region output -------------------------------------
//     for (int i = 0; i < side_square; ++i)
//     {
//         int row = i / side;
//         int col = i % side;
//         for (int n = 0; n < params.num; ++n)
//         {
//             int obj_index = EntryIndex(side, params.coords, params.classes, n * side * side + i, params.coords);
//             int box_index = EntryIndex(side, params.coords, params.classes, n * side * side + i, 0);
//             float scale = output_blob[obj_index];
//             if (scale < threshold)
//                 continue;
//             double x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w;
//             double y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h;
//             double height = std::exp(output_blob[box_index + 3 * side_square]) * params.anchors[2 * n + 1];
//             double width = std::exp(output_blob[box_index + 2 * side_square]) * params.anchors[2 * n];
//             for (int j = 0; j < params.classes; ++j)
//             {
//                 int class_index = EntryIndex(side, params.coords, params.classes, n * side_square + i, params.coords + 1 + j);
//                 float prob = scale * output_blob[class_index];
//                 if (prob < threshold)
//                     continue;
//                 DetectionObject obj(x, y, height, width, j, prob,
//                                     static_cast<float>(original_im_h) / static_cast<float>(resized_im_h),
//                                     static_cast<float>(original_im_w) / static_cast<float>(resized_im_w));
//                 objects.push_back(obj);
//             }
//         }
//     }
// }

// int main(int argc, char *argv[])
// {

//     /** This demo covers a certain topology and cannot be generalized for any object detection **/
//     std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

//     // read input (video) frame
//     std::string xml_path = "/home/yolov4_tiny_3l/models/yolov4-tiny-3l_best.xml";
//     std::string bin_path = "";
//     std::string image_path = "/home/yolov4_tiny_3l/models/demo.jpg";

//     cv::Mat frame = cv::imread(image_path);

//     cv::Mat next_frame = cv::imread(image_path);
//     size_t height = frame.size().height;
//     size_t width = frame.size().width;

//     Core ie;

//     // ------------------nn-----------------------------------------------------------------------------------

//     // --------------------------- 1. Load inference engine -------------------------------------

//     // --------------- 2. Reading the IR generated by the Model Optimizer (.xml and .bin files) ------------
//     auto cnnNetwork = ie.ReadNetwork(xml_path);

//     // -----------------------------------------------------------------------------------------------------

//     /** YOLOV3-based network should have one input and three output **/
//     // --------------------------- 3. Configuring input and output -----------------------------------------
//     // --------------------------------- Preparing input blobs ---------------------------------------------

//     InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
//     if (inputInfo.size() != 1)
//     {
//         throw std::logic_error("This demo accepts networks that have only one input");
//     }
//     InputInfo::Ptr &input = inputInfo.begin()->second;
//     auto inputName = inputInfo.begin()->first;
//     input->setPrecision(Precision::U8);

//     if (FLAGS_auto_resize)
//     {
//         input->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
//         input->getInputData()->setLayout(Layout::NHWC);
//     }
//     else
//     {
//         input->getInputData()->setLayout(Layout::NCHW);
//     }

//     ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
//     SizeVector &inSizeVector = inputShapes.begin()->second;
//     inSizeVector[0] = 1; // set batch to 1
//     cnnNetwork.reshape(inputShapes);
//     // --------------------------------- Preparing output blobs -------------------------------------------

//     OutputsDataMap outputInfo(cnnNetwork.getOutputsInfo());
//     for (auto &output : outputInfo)
//     {
//         output.second->setPrecision(Precision::FP32);
//         output.second->setLayout(Layout::NCHW);
//     }
//     // -----------------------------------------------------------------------------------------------------

//     // --------------------------- 4. Loading model to the device ------------------------------------------

//     ExecutableNetwork network = ie.LoadNetwork(cnnNetwork, "CPU");

//     // -----------------------------------------------------------------------------------------------------

//     // --------------------------- 5. Creating infer request -----------------------------------------------
//     InferRequest::Ptr async_infer_request_next = network.CreateInferRequestPtr();
//     InferRequest::Ptr async_infer_request_curr = network.CreateInferRequestPtr();
//     // -----------------------------------------------------------------------------------------------------

//     // --------------------------- 6. Doing inference ------------------------------------------------------

//     bool isLastFrame = false;
//     bool isAsyncMode = false;   // execution is always started using SYNC mode
//     bool isModeChanged = false; // set to TRUE when execution mode is changed (SYNC<->ASYNC)

//     typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
//     auto total_t0 = std::chrono::high_resolution_clock::now();
//     auto wallclock = std::chrono::high_resolution_clock::now();
//     double ocv_render_time = 0;

//     // std::cout << "To close the application, press 'CTRL+C' here or switch to the output window and press ESC key" << std::endl;
//     // std::cout << "To switch between sync/async modes, press TAB key in the output window" << std::endl;
//     // cv::Size graphSize{static_cast<int>(width / 4), 60};
//     // Presenter presenter(FLAGS_u, static_cast<int>(height) - graphSize.height - 10, graphSize);

//     auto t0 = std::chrono::high_resolution_clock::now();
//     // Here is the first asynchronous point:
//     // in the Async mode, we capture frame to populate the NEXT infer request
//     // in the regular mode, we capture frame to the CURRENT infer request
//     if (isAsyncMode)
//     {
//         if (isModeChanged)
//         {
//             FrameToBlob(frame, async_infer_request_curr, inputName);
//         }
//         if (!isLastFrame)
//         {
//             FrameToBlob(next_frame, async_infer_request_next, inputName);
//         }
//     }
//     else if (!isModeChanged)
//     {
//         FrameToBlob(frame, async_infer_request_curr, inputName);
//     }
//     auto t1 = std::chrono::high_resolution_clock::now();
//     double ocv_decode_time = std::chrono::duration_cast<ms>(t1 - t0).count();

//     t0 = std::chrono::high_resolution_clock::now();
//     // Main sync point:
//     // in the true Async mode, we start the NEXT infer request while waiting for the CURRENT to complete
//     // in the regular mode, we start the CURRENT request and wait for its completion
//     if (isAsyncMode)
//     {
//         if (isModeChanged)
//         {
//             async_infer_request_curr->StartAsync();
//         }
//         if (!isLastFrame)
//         {
//             async_infer_request_next->StartAsync();
//         }
//     }
//     else if (!isModeChanged)
//     {
//         async_infer_request_curr->StartAsync();
//     }

//     if (OK == async_infer_request_curr->Wait(IInferRequest::WaitMode::RESULT_READY))
//     {

//         // ---------------------------Processing output blobs--------------------------------------------------
//         // Processing results of the CURRENT request
//         const TensorDesc &inputDesc = inputInfo.begin()->second.get()->getTensorDesc();
//         unsigned long resized_im_h = 288; //getTensorHeight(inputDesc);
//         unsigned long resized_im_w = 288; //getTensorWidth(inputDesc);
//         std::vector<DetectionObject>
//             objects;
//         // Parsing outputs
//         for (auto &output : outputInfo)
//         {
//             auto output_name = output.first;
//             Blob::Ptr blob = async_infer_request_curr->GetBlob(output_name);
//             ParseYOLOV3Output(cnnNetwork, output_name, blob, resized_im_h, resized_im_w, height, width, 0.1, objects);
//         }
//         // Filtering overlapping boxes
//         std::sort(objects.begin(), objects.end(), std::greater<DetectionObject>());
//         for (size_t i = 0; i < objects.size(); ++i)
//         {
//             if (objects[i].confidence == 0)
//                 continue;
//             for (size_t j = i + 1; j < objects.size(); ++j)
//                 if (IntersectionOverUnion(objects[i], objects[j]) >= 0.4)
//                     objects[j].confidence = 0;
//         }
//         // Drawing boxes
//         for (auto &object : objects)
//         {
//             if (object.confidence < 0.1)
//                 continue;
//             auto label = object.class_id;
//             float confidence = object.confidence;
//             if (confidence > 0.1)
//             {
//                 /** Drawing only objects when >confidence_threshold probability **/
//                 cv::rectangle(frame, cv::Point2f(static_cast<float>(object.xmin), static_cast<float>(object.ymin)),
//                               cv::Point2f(static_cast<float>(object.xmax), static_cast<float>(object.ymax)), cv::Scalar(0, 0, 255));
//             }
//         }
//         cv::imwrite("/home/yolov4_tiny_3l/models/demo2.jpg", frame);
//     }

//     return 0;
// }
