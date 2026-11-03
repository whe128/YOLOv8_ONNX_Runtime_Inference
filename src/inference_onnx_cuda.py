import onnxruntime as ort
import cv2
import numpy as np
import time


print("ONNX Runtime:", ort.__version__)

# preload DLLs to ensure that the CUDA provider is loaded correctly
ort.preload_dlls()

# print cuda cuDNN dependency information
# ort.print_debug_info()

print("Available Providers:", ort.get_available_providers())

session = ort.InferenceSession(
    "models/yolov8n.onnx",
    providers=["CUDAExecutionProvider"]  # cpu or cuda
)

# get the providers available in the session
print("Providers:", session.get_providers())

input_info = session.get_inputs()[0]
output_info = session.get_outputs()[0]

print("Input:", input_info.name, input_info.shape, input_info.type)
print("Output:", output_info.name, output_info.shape, output_info.type)


def preprocess(image, input_size = 640):
    h, w = image.shape[:2]
    scale = min(input_size / h, input_size / w)

    new_w = round(w * scale)
    new_h = round(h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    pad_w = input_size - new_w
    pad_h = input_size - new_h

    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )

    # Convert BGR to RGB
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

    # original image 0~255, uint8 type
    # uniformly scale to 0~1, float32 type
    rgb_unify = rgb.astype(np.float32) / 255.0

    # HWC to CHW
    tensor = np.transpose(rgb_unify, (2, 0, 1))

    # CHW to NCHW
    tensor = np.expand_dims(tensor, axis=0)

    return np.ascontiguousarray(tensor), scale, left, top


# =========================
# 3. Inference
# =========================
# image = cv2.imread("images/car.png")

image = cv2.imread("images/road.webp")


if image is None:
    print("Failed to read image")
    exit(1)

# CPU preprocessing
input_tensor, scale, left, top = preprocess(image)

# Allocate GPU memory and copy input tensor to GPU
gpu_input = ort.OrtValue.ortvalue_from_numpy(
    input_tensor,
    device_type="cuda",
    device_id=0
)

# create IO binding
io_binding = session.io_binding()

# bind input directly to GPU memory
io_binding.bind_ortvalue_input(
    input_info.name,
    gpu_input
)

# bind output directly to GPU memory
io_binding.bind_output(
    output_info.name,
    device_type="cuda",
    device_id=0
)

# start_time = time.perf_counter()

# session.run_with_iobinding(io_binding)

# end_time = time.perf_counter()

# print(f"Inference time: {(end_time-start_time)*1000:.2f} ms")

# outputs = io_binding.get_outputs()






# =========================
# Warm-up
# =========================

print("Warm-up...")

for i in range(20):
    session.run_with_iobinding(io_binding)
print("Warm-up completed")


# =========================
# Benchmark
# =========================

times = []

for i in range(100):
    start = time.perf_counter()

    session.run_with_iobinding(io_binding)

    end = time.perf_counter()

    times.append((end - start) * 1000)

print(f"Average Inference time: {np.mean(times):.2f} ms")


outputs = io_binding.get_outputs()
outputs = outputs[0].numpy()  # copy output from GPU to CPU

# yolov8 output: [1, 84, 8400]
predictions = outputs[0]  # shape: (84, 8400)

# [84, 8400] -> [8400, 84]
predictions = np.transpose(predictions, (1, 0))



# =========================
# 4. Decode + Confidence Filter
# =========================
# 8400 anchors and each anchor has 84 values: [x, y, w, h, conf, cls0, cls1, ..., cls79]
# need to filter out anchors with low confidence and decode the bounding boxes


boxes = []
scores = []
class_ids = []

conf_threshold = 0.25
# intersection over union, if overlap to much that over 0.45,
# delete the box with lower confidence
# iou = area(a & b) / area(a|b)
iou_threshold = 0.45

for pred in predictions:
    cx = pred[0]
    cy = pred[1]
    w = pred[2]
    h = pred[3]


    confidence = pred[4]
    class_id = 0

    for i in range(1, len(pred) - 4):
        if pred[4 + i] > confidence:
            confidence = pred[4 + i]
            class_id = i

    # drop low confidence predictions
    if confidence < conf_threshold:
        continue

    # xywh to xyxy
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # remove padding and scale back to original image size
    x1 = (x1 - left) / scale
    y1 = (y1 - top) / scale
    x2 = (x2 - left) / scale
    y2 = (y2 - top) / scale


    image_h = image.shape[0]
    image_w = image.shape[1]


    # clip the box to be inside the image
    # let value not less than 0 and not greater than image size

    x1 = np.clip(x1, 0, image_w - 1)
    y1 = np.clip(y1, 0, image_h - 1)
    x2 = np.clip(x2, 0, image_w - 1)
    y2 = np.clip(y2, 0, image_h - 1)

    boxes.append([
        float(x1), float(y1), float(x2 - x1), float(y2 - y1)
    ])

    scores.append(confidence)
    class_ids.append(class_id)

# =========================
# 5. Non-Maximum Suppression
# =========================

indices = []

for cls in set(class_ids):
    # fix the class id, and deal with the duplicate

    # get all index of this class
    cls_indices = [
        i for i, c in enumerate(class_ids)
        if c == cls
    ]

    cls_boxes = [boxes[i] for i in cls_indices]
    cls_scores = [scores[i] for i in cls_indices]

    keep = cv2.dnn.NMSBoxes(
        cls_boxes,
        cls_scores,
        conf_threshold,
        iou_threshold
    )

    for k in np.array(keep).flatten():
        indices.append(cls_indices[k])



# =========================
# 6. Draw Results
# =========================

class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife"
]

for i in indices:
    x, y, w, h = map(int, boxes[i])
    class_id = class_ids[i]
    score = scores[i]

    label = f"{class_names[class_id]}: {score:.2f}"

    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    cv2.putText(
        image,
        label,
        (x, max(y-10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )

    print(
        f"Class ID: {cls}, "
        f"Confidence: {score:.3f}, "
        f"BBox: {x, y, w, h}"
    )

cv2.imshow("Result", image)
cv2.waitKey(0)

