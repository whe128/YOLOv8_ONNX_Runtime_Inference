from ultralytics import YOLO
import time

# 1. load yolo model
model = YOLO("./models/yolov8n.pt")

results = model.predict(
    source="images/car.png",
    imgsz=640,
    conf=0.25,
    device="cuda",  # GPU if not cuda, use CPU
    save=False
)

# 2. run the inference
start = time.perf_counter()

results = model.predict(
    source="images/car.png",
    imgsz=640,
    conf=0.25,
    device="cuda",  # GPU if not cuda, use CPU
    save=False
)

end = time.perf_counter()

print(f"Inference time: {(end - start) * 1000:.2f} ms")

# 3. output the inference result
for result in results:
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()

        print(
            f"Class: {model.names[cls]}, "
            f"Confidence: {conf:.3f}, "
            f"BBox: {xyxy}"
        )
