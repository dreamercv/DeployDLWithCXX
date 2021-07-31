# OpenVINO-YOLOV4-tiny-3l

## Environment

Tensorflow 1.15.5

Training：https://github.com/AlexeyAB/darknet


## How to use

Train your yolov4-tiny-3l model first

```shell
#weights- > pb

python convert_weights_pb.py --class_names cfg/coco.names --weights_file yolov4-tiny-3l.weights --data_format NHWC

#pb -> ir
python mo.py --input_model frozen_darknet_yolov4_model.pb --transformations_config yolov4.json --batch 1 --reverse_input_channels

```

Tips:

1. python demo for different OpenVINO version:https://github.com/TNTWEN/OpenVINO-YOLOV4/tree/master/pythondemo

2. Compile C++ demo by yourself:

   \Intel\openvino_2021.3.394\deployment_tools\open_model_zoo\demos\multi_channel_object_detection_demo_yolov3\cpp

3. How to use custom model:

   (1)  When running convert_weights_pb.py use your .names file

   (2)  Modify "classes" in yolov4.json

