// Streaming YOLOv8 inference with perspective-aware tracking.
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {
constexpr int kInputSize = 640;
constexpr int kNumClasses = 80;
constexpr int kNumAnchors = 8400;
constexpr int kQueueMaxSize = 10;
constexpr int kMaxLostCount = 5;
constexpr float kConfidenceThreshold = 0.5f;
constexpr float kIouThreshold = 0.6f;
const std::string kModelPath = "models/yolov8n.onnx";
const std::string kVideoPath = "images/test_1.mp4";

const std::vector<std::string> kClassNames = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};
const std::array<cv::Scalar, 11> kColors = {
    cv::Scalar(0, 0, 255), cv::Scalar(0, 200, 0), cv::Scalar(255, 0, 0),
    cv::Scalar(0, 180, 180), cv::Scalar(180, 0, 180), cv::Scalar(180, 120, 0),
    cv::Scalar(60, 60, 180), cv::Scalar(180, 60, 60), cv::Scalar(80, 140, 80),
    cv::Scalar(140, 80, 140), cv::Scalar(80, 80, 140)};

struct LetterboxInfo { float scale; int left; int top; };
struct Box { float x; float y; float width; float height; };
struct Detection { Box box; float score; int class_id; };
struct Track {
    int id;
    int class_id;
    float score;
    Box box;
    double timestamp;
    float vx;
    float vy;
    int lost_count;
    std::vector<std::pair<double, Box>> history;
};
struct FramePacket { cv::Mat frame; double timestamp; };

class FrameQueue {
public:
    void push(FramePacket packet) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (frames_.size() == kQueueMaxSize) {
            frames_.pop();
            ++dropped_;
        }
        frames_.push(std::move(packet));
        condition_.notify_one();
    }

    bool pop(FramePacket& packet, const std::atomic<bool>& stop) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait_for(lock, std::chrono::seconds(1),
                            [&] { return !frames_.empty() || stop.load(); });
        if (frames_.empty()) return false;
        packet = std::move(frames_.front());
        frames_.pop();
        return true;
    }

    void wake() { condition_.notify_all(); }
    int dropped() const { return dropped_.load(); }

private:
    std::queue<FramePacket> frames_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::atomic<int> dropped_{0};
};

std::vector<float> preprocess(const cv::Mat& image, LetterboxInfo& info) {
    const float scale = std::min(static_cast<float>(kInputSize) / image.rows,
                                 static_cast<float>(kInputSize) / image.cols);
    const int width = static_cast<int>(std::lround(image.cols * scale));
    const int height = static_cast<int>(std::lround(image.rows * scale));
    cv::Mat resized, padded, rgb;
    cv::resize(image, resized, cv::Size(width, height));
    const int pad_width = kInputSize - width;
    const int pad_height = kInputSize - height;
    const int left = pad_width / 2;
    const int top = pad_height / 2;
    cv::copyMakeBorder(resized, padded, top, pad_height - top, left,
                       pad_width - left, cv::BORDER_CONSTANT,
                       cv::Scalar(114, 114, 114));
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

    std::vector<float> tensor(3 * kInputSize * kInputSize);
    for (int y = 0; y < kInputSize; ++y) {
        const auto* row = rgb.ptr<cv::Vec3f>(y);
        for (int x = 0; x < kInputSize; ++x)
            for (int channel = 0; channel < 3; ++channel)
                tensor[channel * kInputSize * kInputSize + y * kInputSize + x] =
                    row[x][channel];
    }
    info = {scale, left, top};
    return tensor;
}

float iou(const Box& a, const Box& b) {
    const float x1 = std::max(a.x, b.x), y1 = std::max(a.y, b.y);
    const float x2 = std::min(a.x + a.width, b.x + b.width);
    const float y2 = std::min(a.y + a.height, b.y + b.height);
    const float intersection = std::max(0.0f, x2 - x1) *
                               std::max(0.0f, y2 - y1);
    const float total = a.width * a.height + b.width * b.height - intersection;
    return total > 0.0f ? intersection / total : 0.0f;
}

std::vector<Detection> decode(const float* output, const cv::Mat& image,
                              const LetterboxInfo& info) {
    std::vector<Detection> detections;
    for (int anchor = 0; anchor < kNumAnchors; ++anchor) {
        const float cx = output[anchor];
        const float cy = output[kNumAnchors + anchor];
        const float width = output[2 * kNumAnchors + anchor];
        const float height = output[3 * kNumAnchors + anchor];
        float score = output[4 * kNumAnchors + anchor];
        int class_id = 0;
        for (int cls = 1; cls < kNumClasses; ++cls) {
            const float candidate = output[(4 + cls) * kNumAnchors + anchor];
            if (candidate > score) { score = candidate; class_id = cls; }
        }
        if (score < kConfidenceThreshold) continue;
        float x1 = (cx - width / 2.0f - info.left) / info.scale;
        float y1 = (cy - height / 2.0f - info.top) / info.scale;
        float x2 = (cx + width / 2.0f - info.left) / info.scale;
        float y2 = (cy + height / 2.0f - info.top) / info.scale;
        x1 = std::clamp(x1, 0.0f, static_cast<float>(image.cols - 1));
        y1 = std::clamp(y1, 0.0f, static_cast<float>(image.rows - 1));
        x2 = std::clamp(x2, 0.0f, static_cast<float>(image.cols - 1));
        y2 = std::clamp(y2, 0.0f, static_cast<float>(image.rows - 1));
        if (x2 > x1 && y2 > y1)
            detections.push_back({{x1, y1, x2 - x1, y2 - y1}, score, class_id});
    }
    return detections;
}

std::vector<int> nms(const std::vector<Detection>& detections) {
    std::vector<int> kept;
    for (int cls = 0; cls < kNumClasses; ++cls) {
        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<int> source;
        for (int i = 0; i < static_cast<int>(detections.size()); ++i) {
            if (detections[i].class_id != cls) continue;
            const Box& b = detections[i].box;
            boxes.emplace_back(static_cast<int>(b.x), static_cast<int>(b.y),
                               static_cast<int>(b.width), static_cast<int>(b.height));
            scores.push_back(detections[i].score);
            source.push_back(i);
        }
        std::vector<int> local;
        cv::dnn::NMSBoxes(boxes, scores, kConfidenceThreshold, kIouThreshold, local);
        for (int index : local) kept.push_back(source[index]);
    }
    return kept;
}

cv::Point2f transform_point(const cv::Mat& transform, cv::Point2f point) {
    std::vector<cv::Point2f> source{point}, destination;
    cv::perspectiveTransform(source, destination, transform);
    return destination.front();
}

float predicted_iou(const Track& track, const Box& current, double dt,
                    const cv::Mat& pixel_to_road, const cv::Mat& road_to_pixel) {
    const cv::Point2f last_foot(track.box.x + track.box.width / 2.0f,
                                track.box.y + track.box.height);
    cv::Point2f road = transform_point(pixel_to_road, last_foot);
    road.x += track.vx * static_cast<float>(dt);
    road.y += track.vy * static_cast<float>(dt);
    const cv::Point2f foot = transform_point(road_to_pixel, road);
    const Box predicted{foot.x - track.box.width / 2.0f,
                        foot.y - track.box.height, track.box.width, track.box.height};
    return iou(predicted, current);
}

// Returns minimum-cost one-to-one assignments for a rectangular matrix.
std::vector<std::pair<int, int>> hungarian(
    const std::vector<std::vector<float>>& costs) {
    if (costs.empty() || costs.front().empty()) return {};
    const int rows = static_cast<int>(costs.size());
    const int cols = static_cast<int>(costs.front().size());
    const bool transpose = rows > cols;
    const int n = transpose ? cols : rows, m = transpose ? rows : cols;
    std::vector<float> u(n + 1), v(m + 1);
    std::vector<int> p(m + 1), way(m + 1);
    for (int row = 1; row <= n; ++row) {
        p[0] = row;
        int column0 = 0;
        std::vector<float> min_value(m + 1, std::numeric_limits<float>::max());
        std::vector<bool> used(m + 1, false);
        do {
            used[column0] = true;
            const int row0 = p[column0];
            float delta = std::numeric_limits<float>::max();
            int column1 = 0;
            for (int column = 1; column <= m; ++column) {
                if (used[column]) continue;
                const float value = transpose ? costs[column - 1][row0 - 1]
                                              : costs[row0 - 1][column - 1];
                const float current = value - u[row0] - v[column];
                if (current < min_value[column]) {
                    min_value[column] = current;
                    way[column] = column0;
                }
                if (min_value[column] < delta) {
                    delta = min_value[column];
                    column1 = column;
                }
            }
            for (int column = 0; column <= m; ++column) {
                if (used[column]) { u[p[column]] += delta; v[column] -= delta; }
                else min_value[column] -= delta;
            }
            column0 = column1;
        } while (p[column0] != 0);
        do {
            const int previous = way[column0];
            p[column0] = p[previous];
            column0 = previous;
        } while (column0 != 0);
    }
    std::vector<std::pair<int, int>> result;
    for (int column = 1; column <= m; ++column) {
        if (p[column] == 0) continue;
        result.emplace_back(transpose ? column - 1 : p[column] - 1,
                            transpose ? p[column] - 1 : column - 1);
    }
    return result;
}

void update_tracks(std::vector<Track>& tracks, int& next_id,
                   const std::vector<Detection>& detections,
                   const std::vector<int>& kept, double timestamp,
                   const cv::Mat& pixel_to_road, const cv::Mat& road_to_pixel) {
    std::vector<std::vector<float>> costs(
        kept.size(), std::vector<float>(tracks.size(), 1e6f));
    for (int detection_position = 0; detection_position < static_cast<int>(kept.size());
         ++detection_position) {
        const Detection& detection = detections[kept[detection_position]];
        for (int track_position = 0; track_position < static_cast<int>(tracks.size());
             ++track_position) {
            const Track& track = tracks[track_position];
            const double dt = timestamp - track.timestamp;
            if (dt <= 0.0) continue;
            const float overlap = predicted_iou(track, detection.box, dt,
                                                pixel_to_road, road_to_pixel);
            if (overlap >= 0.1f)
                costs[detection_position][track_position] =
                    1.0f - overlap +
                    (track.class_id == detection.class_id ? 0.0f : 0.3f);
        }
    }
    std::vector<bool> matched_detections(kept.size()), matched_tracks(tracks.size());
    for (const auto& match : hungarian(costs)) {
        const int detection_position = match.first, track_position = match.second;
        if (costs[detection_position][track_position] >= 1e5f) continue;
        matched_detections[detection_position] = true;
        matched_tracks[track_position] = true;
        Track& track = tracks[track_position];
        const Detection& detection = detections[kept[detection_position]];
        const double dt = timestamp - track.timestamp;
        const cv::Point2f current_foot(
            detection.box.x + detection.box.width / 2.0f,
            detection.box.y + detection.box.height);
        const cv::Point2f previous_foot(
            track.box.x + track.box.width / 2.0f, track.box.y + track.box.height);
        const cv::Point2f current_road =
            transform_point(pixel_to_road, current_foot);
        const cv::Point2f previous_road =
            transform_point(pixel_to_road, previous_foot);
        track.vx = (current_road.x - previous_road.x) / static_cast<float>(dt);
        track.vy = (current_road.y - previous_road.y) / static_cast<float>(dt);
        track.class_id = detection.class_id;
        track.score = detection.score;
        track.box = detection.box;
        track.timestamp = timestamp;
        track.lost_count = 0;
        track.history.emplace_back(timestamp, detection.box);
    }
    for (int i = 0; i < static_cast<int>(tracks.size()); ++i)
        if (!matched_tracks[i]) ++tracks[i].lost_count;
    for (int i = 0; i < static_cast<int>(kept.size()); ++i) {
        if (matched_detections[i]) continue;
        const Detection& detection = detections[kept[i]];
        tracks.push_back({next_id++, detection.class_id, detection.score, detection.box,
                          timestamp, 0.0f, 0.0f, 0, {}});
    }
    tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
                                [](const Track& track) {
                                    return track.lost_count > kMaxLostCount;
                                }),
                 tracks.end());
}

void draw_tracks(cv::Mat& image, const std::vector<Track>& tracks) {
    for (const Track& track : tracks) {
        if (track.lost_count > 0) continue;
        const cv::Scalar color = kColors[track.id % kColors.size()];
        const int x = static_cast<int>(track.box.x), y = static_cast<int>(track.box.y);
        const int width = static_cast<int>(track.box.width);
        const int height = static_cast<int>(track.box.height);
        std::vector<cv::Point> centers;
        const size_t start = track.history.size() > 10 ? track.history.size() - 10 : 0;
        for (size_t i = start; i < track.history.size(); ++i) {
            const Box& box = track.history[i].second;
            centers.emplace_back(static_cast<int>(box.x + box.width / 2.0f),
                                 static_cast<int>(box.y + box.height / 2.0f));
        }
        if (centers.size() >= 2) cv::polylines(image, centers, false, color, 1);
        cv::rectangle(image, cv::Point(x, y), cv::Point(x + width, y + height),
                      color, 1);
        const float speed = std::sqrt(track.vx * track.vx + track.vy * track.vy) *
                            3.6f * 2.0f;
        cv::putText(image,
                    "[" + kClassNames[track.class_id] + "] " +
                        cv::format("%.1f km/h", speed),
                    cv::Point(x, std::max(y - 10, 20)),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, color, 2);
    }
}

void produce_frames(FrameQueue& queue, std::atomic<bool>& stop) {
    cv::VideoCapture capture;
    if (!capture.open(kVideoPath, cv::CAP_FFMPEG)) {
        capture.release();
        if (!capture.open(kVideoPath, cv::CAP_MSMF)) {
            capture.release();
            if (!capture.open(kVideoPath, cv::CAP_DSHOW)) {
                capture.release();
                capture.open(kVideoPath, cv::CAP_ANY);
            }
        }
    }
    if (!capture.isOpened()) {
        std::cerr << "Failed to open video: " << kVideoPath
                  << ". OpenCV backends tried: FFmpeg, Media Foundation, "
                     "DirectShow, default.\n";
        stop.store(true); queue.wake(); return;
    }
    std::cout << "Video backend: " << capture.getBackendName() << '\n';
    const double fps = capture.get(cv::CAP_PROP_FPS);
    if (fps <= 0.0) {
        std::cerr << "Video has an invalid FPS\n";
        stop.store(true); queue.wake(); return;
    }
    std::cout << "Video FPS: " << fps << '\n';
    const auto interval = std::chrono::duration<double>(1.0 / fps);
    while (!stop.load()) {
        const auto start = std::chrono::steady_clock::now();
        cv::Mat frame;
        if (!capture.read(frame)) {
            std::cout << "End of video\n";
            stop.store(true); queue.wake(); break;
        }
        const double timestamp = std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        queue.push({std::move(frame), timestamp});
        const auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed < interval) std::this_thread::sleep_for(interval - elapsed);
    }
}
}  // namespace

int main() {
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolov8_stream");
        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
#ifdef USE_CUDA
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.device_id = 0;
        options.AppendExecutionProvider_CUDA(cuda_options);
#endif
        const std::wstring model_path(kModelPath.begin(), kModelPath.end());
        Ort::Session session(env, model_path.c_str(), options);
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name = session.GetInputNameAllocated(0, allocator);
        auto output_name = session.GetOutputNameAllocated(0, allocator);
        const auto shape = session.GetOutputTypeInfo(0)
                               .GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() != 3 || shape[0] != 1 ||
            shape[1] != kNumClasses + 4 || shape[2] != kNumAnchors)
            throw std::runtime_error("Unexpected YOLO output shape");
        std::cout << "Input: " << input_name.get() << "\nOutput: "
                  << output_name.get() << '\n';

        const std::array<int64_t, 4> input_shape = {1, 3, kInputSize, kInputSize};
        const Ort::MemoryInfo memory_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const char* input_names[] = {input_name.get()};
        const char* output_names[] = {output_name.get()};


        auto run = [&](std::vector<float>& data) {
            auto tensor = Ort::Value::CreateTensor<float>(
                memory_info, data.data(), data.size(), input_shape.data(),
                input_shape.size());
            return session.Run(Ort::RunOptions{nullptr}, input_names, &tensor, 1, output_names, 1);
        };

        cv::Mat dummy(kInputSize, kInputSize, CV_8UC3, cv::Scalar(0, 0, 0));
        LetterboxInfo dummy_info;
        std::vector<float> dummy_data = preprocess(dummy, dummy_info);
        std::cout << "Warmup...\n";
        for (int i = 0; i < 10; ++i) run(dummy_data);
        std::cout << "Warmup done.\n";

        cv::Mat pixel_to_road = cv::getPerspectiveTransform(
            std::vector<cv::Point2f>{{478, 94}, {797, 100}, {133, 487}, {1204, 487}},
            std::vector<cv::Point2f>{{0, 0}, {18.75f, 0}, {0, 22.5f}, {18.75f, 22.5f}});
        cv::Mat road_to_pixel = pixel_to_road.inv();
        FrameQueue frame_queue;
        std::atomic<bool> stop{false};
        std::thread producer(produce_frames, std::ref(frame_queue), std::ref(stop));
        std::vector<Track> tracks;
        int next_id = 0, processed = 0;
        const auto total_start = std::chrono::steady_clock::now();
        while (!stop.load()) {
            FramePacket packet;
            if (!frame_queue.pop(packet, stop)) continue;
            LetterboxInfo info;
            std::vector<float> input = preprocess(packet.frame, info);
            auto outputs = run(input);
            const auto* output = outputs.front().GetTensorData<float>();
            const auto detections = decode(output, packet.frame, info);
            const auto kept = nms(detections);
            update_tracks(tracks, next_id, detections, kept, packet.timestamp,
                          pixel_to_road, road_to_pixel);
            draw_tracks(packet.frame, tracks);
            cv::imshow("Result", packet.frame);
            if (cv::waitKey(1) == 27) {
                stop.store(true); frame_queue.wake();
            }
            ++processed;
        }
        stop.store(true); frame_queue.wake(); producer.join();
        cv::destroyAllWindows();
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - total_start).count();
        std::cout << "Total frames processed: " << processed << '\n'
                  << "Total time: " << std::fixed << std::setprecision(2)
                  << seconds << " seconds\nAverage FPS: "
                  << (seconds > 0 ? processed / seconds : 0) << '\n'
                  << "Dropped frames: " << frame_queue.dropped() << '\n';
    } catch (const Ort::Exception& error) {
        std::cerr << "ONNX Runtime exception: " << error.what() << '\n';
        return EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
