# YOLOv8 ONNX Runtime Inference
https://github.com/user-attachments/assets/ead40b2d-a201-431d-b860-43ef52df7549

This project demonstrates an end-to-end edge-AI deployment workflow for a
YOLOv8 object-detection model:
1. Export a PyTorch YOLOv8 model to ONNX.
2. Run ONNX inference with ONNX Runtime on CPU or CUDA.
3. Implement image preprocessing and YOLO postprocessing with OpenCV.
4. Deploy the detector in Python and C++.
5. Process a video stream with a low-latency producer/consumer pipeline.
6. Track objects using perspective-aware motion prediction and Hungarian
   assignment.
7. Experiment with vehicle ReID features as an alternative association signal.

The default model is `models/yolov8n.onnx`, exported at an input resolution of
`640 x 640`.

## Project layout

```text
model_inference/
├── images/
│   ├── car.png
│   ├── road.webp
│   └── test_1.mp4
├── models/
│   ├── yolov8n.pt
│   ├── yolov8n.onnx
│   └── osnet_ain_x1_0_vehicle_reid.onnx
├── src/
│   ├── export.py
│   ├── inference.py
│   ├── inference.cpp
│   ├── inference_onnx.py
│   ├── inference_onnx_cuda.py
│   ├── inference_onnx_stream.py
│   ├── inference_onnx_stream.cpp
│   └── inference_onnx_stream_reid.py
├── CMakeLists.txt
└── requirements.txt
```

## Environment

The project was developed and tested on Windows with:

- Python
- PyTorch and Ultralytics YOLO
- OpenCV
- ONNX Runtime with the CUDA execution provider
- CUDA and cuDNN
- CMake
- Visual Studio C++ build tools

The C++ build uses C++17.

## 1. Export YOLOv8 to ONNX

The export script loads the PyTorch model and exports an ONNX model with
opset 17 and a simplified graph:

```bash
python src/export.py
```

The relevant export configuration is:

```python
model.export(
    format="onnx",
    imgsz=640,
    opset=17,
    simplify=True,
)
```

The generated model is copied or saved as:

```text
models/yolov8n.onnx
```

The YOLOv8 output used by this project has the shape:

```text
[1, 84, 8400]
```

The 84 values for each anchor are interpreted as:

```text
[center_x, center_y, width, height, class_0, ..., class_79]
```

The implementation treats the first class score as the initial confidence and
selects the highest class score for the final class ID.

## 2. Image preprocessing

The Python and C++ implementations use the same letterbox preprocessing
pipeline:

1. Read the input image in OpenCV BGR format.
2. Scale the image while preserving its aspect ratio.
3. Resize it so that the largest dimension fits within `640 x 640`.
4. Pad the remaining pixels with the value `(114, 114, 114)`.
5. Convert BGR to RGB.
6. Convert pixel values from `uint8 [0, 255]` to `float32 [0, 1]`.
7. Reorder the tensor from HWC to CHW.
8. Add the batch dimension to produce NCHW.
9. Keep the scale and padding values so output boxes can be mapped back to
   the original image.

The final input tensor shape is:

```text
[1, 3, 640, 640]
```

The tensor is stored contiguously before it is passed to ONNX Runtime.

## 3. YOLO postprocessing

Postprocessing is implemented with OpenCV and contains two stages.

### Confidence filtering and box decoding

For every one of the 8400 anchors:

1. Select the highest class score.
2. Discard detections below the configured confidence threshold.
3. Convert center-based boxes:

```text
center_x, center_y, width, height
```

to corner-based boxes:

```text
x1, y1, x2, y2
```

4. Remove the letterbox padding.
5. Divide by the resize scale.
6. Clip the result to the original image dimensions.

### Non-maximum suppression

OpenCV `cv::dnn::NMSBoxes` / `cv2.dnn.NMSBoxes` is run independently for each
class. The final detection result contains:

- Bounding boxes
- Confidence scores
- Class IDs

The single-image Python examples use a confidence threshold of `0.25` and an
IoU threshold of `0.45`. The streaming detector follows the streaming Python
implementation and uses:

```text
confidence threshold = 0.50
IoU threshold        = 0.60
```

## 4. Single-image inference and benchmarking

The project contains several single-image inference examples:

| File                         | Runtime          | Device / purpose                        |
| ---------------------------- | ---------------- | --------------------------------------- |
| `src/inference.py`           | PyTorch          | Baseline PyTorch inference              |
| `src/inference_onnx.py`      | ONNX Runtime     | Python ONNX inference                   |
| `src/inference_onnx_cuda.py` | ONNX Runtime     | CUDA execution provider and I/O binding |
| `src/inference.cpp`          | ONNX Runtime C++ | C++ image inference                     |

The inference tests use warmup iterations before measuring performance. Warmup
is important because the first calls can include graph initialization,
provider initialization, memory allocation, and CUDA kernel setup.

Example Python commands:

```bash
python src/inference.py
python src/inference_onnx.py
python src/inference_onnx_cuda.py
```

The Python ONNX CUDA example uses:

```python
providers=["CUDAExecutionProvider"]
```

The CUDA version also demonstrates ONNX Runtime I/O binding so input and output
can be associated with GPU memory.

## 5. C++ build

The CMake project creates two executables:

```text
yolov8
yolov8_stream
```

Configure with the installed OpenCV package and ONNX Runtime package:

```bash
cmake -S . -B build \
  -DOpenCV_DIR="C:/environment/opencv/build/x64/vc16/lib" \
  -DONNXRUNTIME_ROOT="C:/environment/onnxruntime-win-x64-gpu_cuda13-1.31.1" \
  -DENABLE_CUDA=ON
```

Build the project:

```bash
cmake --build build --config Release -j 4
```

Run the C++ image executable:

```bash
./build/Release/yolov8.exe
```

Run the C++ streaming executable:

```bash
./build/Release/yolov8_stream.exe
```

On Windows, the runtime directory must contain the ONNX Runtime, CUDA, cuDNN,
OpenCV, and OpenCV videoio plugin DLLs. The CMake post-build commands copy
these dependencies when the corresponding paths are configured.

If the video cannot be opened, verify that this file exists beside the
executable's working directory:

```text
opencv_videoio_ffmpeg4100_64.dll
```

The executable expects paths relative to the project root:

```text
models/yolov8n.onnx
images/test_1.mp4
```

Run it from the project root or change the model and video paths in
`src/inference_onnx_stream.cpp`.

## 6. Video streaming pipeline

The streaming implementation is designed for low latency rather than maximum
throughput.

```text
VideoCapture
    ↓
Producer thread
    ↓
Bounded frame queue
    ↓
Preprocess and ONNX inference
    ↓
Decode and NMS
    ↓
Track association
    ↓
Trajectory and speed rendering
```

### Producer thread and frame queue

The producer thread reads frames from `images/test_1.mp4` and attaches a
timestamp to each frame. The queue is bounded to 10 frames. If the queue is
full, the oldest frame is discarded before the newest frame is inserted.

This keeps the displayed result close to the live input position. Without
frame dropping, a slower inference loop would continue processing increasingly
old frames and produce high latency.

The implementation reports the number of dropped frames when it exits.

### Tracking state

Each track stores:

- Track ID
- Class ID
- Confidence score
- Last bounding box
- Last timestamp
- Road-plane velocity `vx` and `vy`
- Lost-frame count
- Recent box history for drawing the trajectory

A track is removed after it has been unmatched for more than five frames.

## 7. Perspective-aware motion prediction

Image-plane pixels do not represent a constant physical distance across the
scene. To estimate motion in road coordinates, the implementation defines four
corresponding points:

```text
Pixel coordinates:
  (478, 94), (797, 100), (133, 487), (1204, 487)

Road coordinates:
  (0, 0), (18.75, 0), (0, 22.5), (18.75, 22.5)
```

OpenCV computes a perspective transform:

```text
pixel position → road position
```

and its inverse:

```text
road position → pixel position
```

For each object, the bottom-center point of the bounding box is used as the
contact point with the road. The previous contact point is transformed to road
coordinates and advanced using velocity and elapsed time:

```text
predicted_road_x = previous_road_x + vx * delta_t
predicted_road_y = previous_road_y + vy * delta_t
```

The predicted road position is then transformed back to image coordinates to
form a predicted bounding box.

The displayed speed is derived from the road-plane velocity and converted to
km/h using the conversion used in the Python implementation.

## 8. IoU and Hungarian tracking

For every current detection and existing track:

1. Predict the track's new box using its previous box, velocity, and elapsed
   time.
2. Calculate IoU between the predicted box and the current detection.
3. Reject pairs with IoU below `0.1`.
4. Use `1 - IoU` as the matching cost.
5. Add a soft class mismatch penalty of `0.3`.
6. Solve the global one-to-one assignment with the Hungarian algorithm.

Matched tracks are updated with the new box, timestamp, velocity, class, and
history. Unmatched tracks increase their lost count. Unmatched detections
create new tracks.

This approach is more stable than matching only by nearest box center because
it uses both object motion and scene geometry.

## 9. ReID experiment

The project also includes a vehicle ReID experiment using:

```text
models/osnet_ain_x1_0_vehicle_reid.onnx
```

The idea is to crop each detected vehicle, extract a feature vector, and
compare feature similarity when associating detections with existing tracks.
This can help recover identity after occlusion or missed detections.

The experiment was substantially slower in the tested configuration:
input: [Batch, Channel, Height, Width]
output: [Batch, 512]

```text
approximately 500–700 ms
```

That latency is too high for the current real-time stream target, so the main
streaming implementation uses perspective-aware IoU and Hungarian matching
instead of ReID on every detection.

Possible future optimizations include:

- Batch ReID crops into one inference call.
- Run ReID only when IoU association is ambiguous.
- Cache embeddings and compare only against candidate tracks.
- Use a smaller ReID model.
- Run detection and ReID asynchronously.

## 10. Observed performance

The measured inference performance was approximately:

| Implementation            | Inference time |
| ------------------------- | -------------: |
| Python streaming pipeline |       20–30 ms |
| C++ streaming pipeline    |       14–17 ms |
| ReID-assisted experiment  |     500–700 ms |

The exact result depends on GPU model, CUDA/cuDNN versions, input resolution,
video decoding, preprocessing, postprocessing, display overhead, and whether
the reported value measures inference only or the complete frame loop.

The C++ implementation is faster in the tested stream because it reduces
Python interpreter overhead and uses native OpenCV and ONNX Runtime APIs.

## 11. What was learned

This project provided practical experience with:

- YOLO model structure, input tensors, and output tensors.
- Exporting a PyTorch model to ONNX.
- Building an ONNX Runtime deployment environment.
- CPU and CUDA execution providers.
- CUDA-accelerated edge inference.
- OpenCV image preprocessing and postprocessing.
- Continuous tensor layout and NCHW model inputs.
- C++ and Python implementations of the same inference pipeline.
- Producer/consumer streaming design.
- Low-latency frame dropping.
- Perspective transforms for approximate real-world motion.
- IoU-based multi-object tracking.
- Hungarian assignment for global detection-to-track matching.
- ReID feature extraction and the cost of naïve per-object inference.
- Measuring performance after model warmup.

The main engineering lesson is that deployment performance is determined by the
whole pipeline, not only model inference. Video decoding, tensor conversion,
memory copies, postprocessing, tracking, feature extraction, and display all
contribute to end-to-end latency.
