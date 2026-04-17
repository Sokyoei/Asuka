#!/bin/bash
# ultralytics cil

# predict
# yolo predict model=yolo11n-seg.pt source=0 imgsz=640

# export
# yolo export model=yolov8s-pose.pt format=onnx opset=11 simplify=True
yolo export model=../models/yolo26l.pt format=engine device=0 dynamic=True simplify=True half=True imgsz=640 batch=32
