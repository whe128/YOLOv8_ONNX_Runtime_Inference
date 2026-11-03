#include <vector>
#include <string>
#include <array>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <stdexcept>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

using namespace std;


// =========================
// 1. Configuration
// =========================
const int INPUT_SIZE = 640; // Input size for the model
const float CONF_THRESHOLD = 0.5; // Confidence threshold for detections
const float IOU_THRESHOLD = 0.45; // IoU threshold for non-max suppression
const int NUM_CLASSES = 80; // Number of classes in the model
const int NUM_ANCHORS = 8400; // Number of anchors per scale


// COCO 80 classes
const vector<string> CLASS_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
};


struct LetterboxInfo{
    float scale;
    int left;
    int top;
};

struct Detection{
    cv::Rect box;
    float confidence;
    int class_id;
};

// =========================
// 2. Preprocessing
// =========================

vector<float> preprocess(const cv::Mat& image, LetterboxInfo& info){
    int h = image.rows;
    int w = image.cols;

    float scale = min(
        static_cast<float>(INPUT_SIZE) / static_cast<float>(h),
        static_cast<float>(INPUT_SIZE) / static_cast<float>(w)
    );

    int new_w = static_cast<int>(w * scale);
    int new_h = static_cast<int>(h * scale);

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h));

    int pad_w = INPUT_SIZE - new_w;
    int pad_h = INPUT_SIZE - new_h;

    // how many pixels to pad on each side
    int left = pad_w / 2;
    int right = pad_w - left;
    int top = pad_h / 2;
    int bottom = pad_h - top;

    cv::Mat padded;
    cv::copyMakeBorder(
        resized, padded, top, bottom, left, right,
        cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114)
    );

    // BGR to RGB
    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

    // uint8 to float32
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

    // HWC to CHW
    // [640, 640, 3] -> [3, 640, 640]

    vector<float> tensor(3*INPUT_SIZE*INPUT_SIZE);

    // onnx and most inference engines expect the input tensor to be
    // in 1D array, so we need to flatten the 3D array
    for(int y=0; y < INPUT_SIZE; y++){
        const cv::Vec3f* row = rgb.ptr<cv::Vec3f>(y);
        for(int x=0; x < INPUT_SIZE; x++){
            for(int c=0; c < 3; c++){
                tensor[c*INPUT_SIZE*INPUT_SIZE + y*INPUT_SIZE + x] = row[x][c];
            }
        }
    }

    info.scale = scale;
    info.left = left;
    info.top = top;

    return tensor;
}


// =========================
// 3. Decode Predictions
// =========================

vector<Detection> decode_predictions(
    const float* output,
    const cv::Mat& image,
    const LetterboxInfo& info
){
    vector<Detection> detections;

    int image_h = image.rows;
    int image_w = image.cols;

    // output shape: [1, 84, 8400] -> [84, 8400]
    // channel 0 cx
    // channel 1 cy
    // channel 2 w
    // channel 3 h
    // channel 4~83: 80 class scores

    for(int a = 0; a < NUM_ANCHORS; a++){
        float cx = output[0*NUM_ANCHORS + a];
        float cy = output[1*NUM_ANCHORS + a];
        float w  = output[2*NUM_ANCHORS + a];
        float h  = output[3*NUM_ANCHORS + a];

        float confidence = output[4*NUM_ANCHORS + a];
        int class_id = 0;

        for(int c = 1; c < NUM_CLASSES; c++){
            float score = output[(4+c)*NUM_ANCHORS + a];
            if(score > confidence){
                class_id = c;
                confidence = score;
            }
        }

        // confidence filtering
        if(confidence < CONF_THRESHOLD){
            continue;
        }

        // XYWH(center) to XYXY
        float x1 = cx - w / 2.0f;
        float y1 = cy - h / 2.0f;
        float x2 = cx + w / 2.0f;
        float y2 = cy + h / 2.0f;

        // remove letterbox padding
        x1 = (x1 - info.left) / info.scale;
        y1 = (y1 - info.top) / info.scale;
        x2 = (x2 - info.left) / info.scale;
        y2 = (y2 - info.top) / info.scale;

        // clip to coordinates
        x1 = clamp(x1, 0.0f, static_cast<float>(image_w - 1));
        y1 = clamp(y1, 0.0f, static_cast<float>(image_h - 1));
        x2 = clamp(x2, 0.0f, static_cast<float>(image_w - 1));
        y2 = clamp(y2, 0.0f, static_cast<float>(image_h - 1));

        int bx = static_cast<int>(round(x1));
        int by = static_cast<int>(round(y1));
        int bw = static_cast<int>(round(x2 - x1));
        int bh = static_cast<int>(round(y2 - y1));

        if(bw <= 0 || bh <= 0){
            continue;
        }

        Detection det;
        det.box = cv::Rect(bx, by, bw, bh);
        det.confidence = confidence;
        det.class_id = class_id;

        detections.push_back(det);
    }

    return detections;
}

// =========================
// 4. Non-Maximum Suppression
// =========================
vector<Detection> apply_nms(const vector<Detection>& detections){
    vector<Detection> results;

    // process each class independently
    for(int cls = 0; cls < NUM_CLASSES; cls++){
        vector<cv::Rect> boxes;
        vector<float> scores;
        vector<int> indices;

        for(int i = 0; i < static_cast<int>(detections.size()); i++){
            if(detections[i].class_id == cls){
                boxes.push_back(detections[i].box);
                scores.push_back(detections[i].confidence);
                indices.push_back(i);
            }
        }

        if(boxes.empty()){
            continue;
        }

        vector<int> keep;

        cv::dnn::NMSBoxes(
            boxes,
            scores,
            CONF_THRESHOLD,
            IOU_THRESHOLD,
            keep
        );

        for(int idx : keep){
            results.push_back(detections[indices[idx]]);
        }
    }
    return results;
}

// =========================
// 5. Draw Results
// =========================
void draw_detections(
    cv::Mat& image,
    const vector<Detection>& detections
){
    for(const auto& det : detections){
        int x = det.box.x;
        int y = det.box.y;
        int w = det.box.width;
        int h = det.box.height;

        string label = CLASS_NAMES[det.class_id] + ": " + to_string(det.confidence);

        cv::rectangle(
            image,
            det.box,
            cv::Scalar(0, 255, 0),
            2
        );

        cv::putText(
            image,
            label,
            cv::Point(x, max(y - 10, 20)),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 255, 0),
            2
        );

        cout << "Class: " << CLASS_NAMES[det.class_id]
            << ", Confidence: " << det.confidence
            << ", BBox: (" << x << ", " << y
            << ", " << w << ", " << h << ")"
            << endl;
    }
}

// =========================
// 6. Main
// =========================

int main(){
    try{
        cout << "ONNX Runtime C++ YOLOv8" << endl;

        // initialize ONNX Runtime
        Ort::Env env(
            OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
            "yolov8"
        );

        Ort::SessionOptions session_options;

        session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_EXTENDED
        );


        // OrtCUDAProviderOptions cuda_options;
        // cuda_options.device_id = 0; // GPU device ID
        // session_options.AppendExecutionProvider_CUDA(cuda_options);


        // CPU Execution Provider
        Ort::Session session(
            env,
            L"models/yolov8n.onnx",
            session_options
        );

        // Get input/output information
        Ort::AllocatorWithDefaultOptions allocator;

        auto input_name = session.GetInputNameAllocated(0, allocator);
        auto output_name = session.GetOutputNameAllocated(0, allocator);

        cout << "Input Name: " << input_name.get() << endl;
        cout << "Output Name: " << output_name.get() << endl;

        // Load image
        cv::Mat image = cv::imread("images/road.webp");

        if(image.empty()){
            cerr << "Failed to read image" << endl;
            return -1;
        }

        // Preprocess
        LetterboxInfo info;
        vector<float> input_data = preprocess(image, info);

        // Create input tensor
        array<int64_t, 4> input_shape = {
            1, 3, INPUT_SIZE, INPUT_SIZE
        };

        Ort::MemoryInfo memory_info =
            Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator,  // in onnxruntime memory pool
                OrtMemTypeDefault   // normal memory
            );

        Ort::Value input_tensor =
            Ort::Value::CreateTensor<float>(
                memory_info,
                input_data.data(),
                input_data.size(),
                input_shape.data(),
                input_shape.size()
            );

        const char* input_names[] = { input_name.get() };
        const char* output_names[] = { output_name.get() };


        vector<double> times;
        times.reserve(100);

        // Warm-up
        cout << "Warm-up..." << endl;

        for (int i = 0; i < 20; i++)
        {
            session.Run(
                Ort::RunOptions{ nullptr },
                input_names,
                &input_tensor,
                1,
                output_names,
                1
            );
        }

        cout << "Warm-up completed" << endl;

        // Benchmark
        for (int i = 0; i < 100; i++)
        {
            auto start = chrono::steady_clock::now();

            session.Run(
                Ort::RunOptions{ nullptr },
                input_names,
                &input_tensor,
                1,
                output_names,
                1
            );

            auto end = chrono::steady_clock::now();

            double elapsed =
                chrono::duration<double, milli>(
                    end - start
                ).count();

            times.push_back(elapsed);
        }

        // Statistics
        sort(times.begin(), times.end());
        double total = 0;
        for (double t : times){
            total += t;
        }

        cout << fixed << setprecision(3);

        cout << "Average: "
            << total / 100 << " ms" << endl;


        // Inference

        auto output_tensors = session.Run(
            Ort::RunOptions{ nullptr },
            input_names,
            &input_tensor,
            1,
            output_names,
            1
        );

        // Get output tensor
        float* output_data = output_tensors[0].GetTensorMutableData<float>();

        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

        cout << "Output shape: [";
        for(auto dim : output_shape){
            cout << dim << ", ";
        }
        cout << "]" << endl;

        if(output_shape.size() != 3 ||
            output_shape[0] != 1 ||
            output_shape[1] != NUM_CLASSES + 4 ||
            output_shape[2] != NUM_ANCHORS){
            cerr << "Unexpected output shape" << endl;
            return -1;
        }

        // Decode predictions
        vector<Detection> detections = decode_predictions(output_data, image, info);

        cout << "Detections before NMS: " << detections.size() << endl;

        // Apply NMS
        vector<Detection> final_detections = apply_nms(detections);
        cout << "Detections after NMS: " << final_detections.size() << endl;

        // Draw results
        draw_detections(image, final_detections);

        cv::imshow("Detections", image);
        cv::waitKey(0);
    }
    catch(const Ort::Exception& e){
        cerr << "ONNX Runtime exception: " << e.what() << endl;
        return -1;
    }
    catch(const exception& e){
        cerr << "Standard exception: " << e.what() << endl;
        return -1;
    }
    catch(...){
        cerr << "Unknown exception occurred" << endl;
        return -1;
    }

    return 0;

}

