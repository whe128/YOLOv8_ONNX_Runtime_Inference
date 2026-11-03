from ultralytics import YOLO

# 1. load yolo model
model = YOLO("./models/yolov8n.pt")

model.export(
    format="onnx",
    imgsz=640,          # input size
    opset=17,
    simplify=True
)
