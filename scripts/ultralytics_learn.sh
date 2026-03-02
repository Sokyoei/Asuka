# ultralytics cil

# predict
# yolo predict model=yolo11n-seg.pt source=0 imgsz=640

# export
# yolo export model=yolov8s-pose.pt format=onnx opset=11 simplify=True
yolo export model=yolov8s-pose.pt format=engine device=0
