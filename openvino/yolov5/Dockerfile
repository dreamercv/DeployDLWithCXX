From  ccr.ccs.tencentyun.com/public_images/cuda10.2-cudnn8.0.4-dev-ubuntu16.04-opencv4.1.1-tensorrt7.1.3-openvino2020r3-workspace:v1.0.0 
RUN rm -rf /home/infer_openvino2020-3_yolov5-v5  && mkdir /home/infer_openvino2020-3_yolov5-v5
COPY . /home/infer_openvino2020-3_yolov5-v5
RUN cd /home/infer_openvino2020-3_yolov5-v5 && rm -rf build && mkdir build && cmake .. && make 
