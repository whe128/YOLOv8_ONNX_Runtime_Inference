import onnxruntime as ort
import cv2
import numpy as np
import time

from scipy.optimize import linear_sum_assignment

import threading
import queue

print("ONNX Runtime:", ort.__version__)

# preload DLLs to ensure that the CUDA provider is loaded correctly
ort.preload_dlls()

# print cuda cuDNN dependency information
# ort.print_debug_info()

print("Available Providers:", ort.get_available_providers())

# ===================== Available =====================
MODEL_PATH = "models/yolov8n.onnx"
VIDEO_PATH = "images/test_1.mp4"
INPUT_SIZE = 640
CONF_THRESH = 0.25
IOU_THRESH = 0.45
# providers = ["CPUExecutionProvider"]
providers = ["CUDAExecutionProvider"]
QUEUE_MAX_SIZE = 10  # low latency, but may drop frames
MAX_LOST_CNT = 12
REID_H, REID_W = 128, 256
REID_DIM = 512

next_track_id = 0
trajectory = []

# traj_element = {
#           id: track_id,
#           class_id: class_id,
#           score: score,
#           last_box: [x, y, w, h],
#           last_ts: ts,
#           vx:
#           vy:
#           feat_bank:  len(set 5)
#           lost_cnt
#           history     []


so = ort.SessionOptions()
so.log_severity_level = 3
session_det = ort.InferenceSession(
    "models/yolov8n.onnx",
    providers = providers
)       # cpu or cuda

session_reid = ort.InferenceSession(
    "models/osnet_ain_x1_0_vehicle_reid.onnx",
    sess_options=so,
    providers = providers
)         # cpu or cuda



# get the providers available in the session
print("Providers:", session_det.get_providers())

input_info = session_det.get_inputs()[0]
output_info = session_det.get_outputs()[0]

print("Input detection:", input_info.name, input_info.shape, input_info.type)
print("Output detection:", output_info.name, output_info.shape, output_info.type)

print("Input reid:", session_reid.get_inputs()[0].name, session_reid.get_inputs()[0].shape, session_reid.get_inputs()[0].type)
print("Output reid:", session_reid.get_outputs()[0].name, session_reid.get_outputs()[0].shape, session_reid.get_outputs()[0].type)

frame_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
stop_event = threading.Event()
drop_cnt = 0

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
# 4. Decode + Confidence Filter
# =========================
# 8400 anchors and each anchor has 84 values: [x, y, w, h, conf, cls0, cls1, ..., cls79]
# need to filter out anchors with low confidence and decode the bounding boxes

def post_process(predictions, scale, left, top, image):

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



    return boxes, scores, class_ids, indices


def frame_producer():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Failed to open video")
        stop_event.set()
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)

    frame_interval = 1.0 / video_fps

    print(f"Video FPS: {video_fps}")

    while not stop_event.is_set():
        t_start = time.perf_counter()
        ret, frame = cap.read()

        if not ret:
            print("End of video")
            stop_event.set()
            break

        ts = time.perf_counter()
        packet = (frame, ts)

        if frame_queue.full():
            try:
                # print("Frame queue is full, dropping frame")
                frame_queue.get_nowait()

                global drop_cnt

                drop_cnt += 1
            except queue.Empty:
                pass


        frame_queue.put(packet)

        elapsed = time.perf_counter() - t_start
        sleep_time = frame_interval - elapsed

        if sleep_time > 0:
            time.sleep(sleep_time)
    cap.release()


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
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

COLOR_PALETTE = [
    (0, 0, 255),    # red
    (0, 200, 0),    # green
    (255, 0, 0),    # blue
    (0, 180, 180),  # cyan
    (180, 0, 180),  # magenta
    (180, 120, 0),  # orange
    (60, 60, 180),  # purple
    (180, 60, 60),  # pink
    (80, 140, 80),  # deep green
    (140, 80, 140), # purple-pink
    (80, 80, 140),  # deep blue
]

def draw_trajectory(image, trajectory):
    for traj in trajectory:
        if traj["lost_cnt"] > 0:
            continue

        box = traj["last_box"]
        x, y, w, h = map(int, box)
        class_id = traj["class_id"]
        score = traj["score"]
        label = f"{class_names[class_id]}: {score:.2f}, ID: {traj['id']}"

        color = COLOR_PALETTE[traj["id"] % len(COLOR_PALETTE)]

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            color,
            1
        )

        cv2.putText(
            image,
            label,
            (x, max(y-10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

    cv2.imshow("Result", image)
    cv2.waitKey(1)


def draw_results(image, boxes, scores, class_ids, indices):
    for i in indices:
        x, y, w, h = map(int, boxes[i])
        class_id = class_ids[i]
        label = class_names[class_id]

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            1
        )

        cv2.putText(
            image,
            label,
            (x, max(y-10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

        # print(
        #     f"Class ID: {class_id}, "
        #     f"Confidence: {score:.3f}, "
        #     f"BBox: {x, y, w, h}"
        # )

    cv2.imshow("Result", image)
    cv2.waitKey(1)

def compute_iou(last_box, box2, vx, vy, dt):
    lx, ly, lw, lh = last_box
    cx_old = lx + lw / 2
    cy_old = ly + lh / 2

    cx_pred = cx_old + vx * dt
    cy_pred = cy_old + vy * dt
    pre_w = lw
    pre_h = lh

    pre_box = [cx_pred - pre_w / 2, cy_pred - pre_h / 2, pre_w, pre_h]

    x1 = max(pre_box[0], box2[0])
    y1 = max(pre_box[1], box2[1])
    x2 = min(pre_box[0] + pre_box[2], box2[0] + box2[2])
    y2 = min(pre_box[1] + pre_box[3], box2[1] + box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = pre_box[2] * pre_box[3] + box2[2] * box2[3] - intersection

    if union == 0:
        return 0.0

    return intersection / union

def extract_reid_feature(image, box, session):
    """
    Crop detection region from frame,
    extract and normalize ReID feature.
    """
    ...
    N = len(boxes)


    if N == 0:
        return np.zeros((0, REID_DIM), dtype=np.float32)

    image_h, image_w = image.shape[:2]

    crops = []

    for b in boxes:
        x, y, w, h = map(int, b)
        x1 = np.clip(x, 0, image_w - 1)
        y1 = np.clip(y, 0, image_h - 1)
        x2 = np.clip(x + w, 0, image_w - 1)
        y2 = np.clip(y + h, 0, image_h - 1)

        if x2 <= x1 or y2 <= y1:
            continue

        # crop -> resize -> normalize -> transpose(HWC->CHW) -> add batch dimension
        c = cv2.resize(image[y1:y2, x1:x2], (128, 256))
        c = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
        c = c.astype(np.float32) / 255.0
        c = np.transpose(c, (2, 0, 1))

        # calculate mean and std for normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        crops.append((c - mean) / std)

    tensor = np.stack(crops).astype(np.float32)     # shape: (N, 3, 256, 128)
    tensor = np.ascontiguousarray(tensor, np.float32)

    out = session.run(
        [session.get_outputs()[0].name],
        {session.get_inputs()[0].name: tensor})[0]  # shape: (N, 512)

    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1.0

    return out / norms  # shape: (N, 512)

def track_detection(trajectory, boxes, scores, class_ids, indices, dets_feats, ts):
    # first check whether the box is already in the trajectory, if yes, update the box and ts
    # if not, add the box to the trajectory

    matched_indices = set()
    matched_dets = set()

    det_indices = list(indices)

    # how many boxes in prediction
    num_dets = len(det_indices)

    # how many trajectory elements
    num_trajs = len(trajectory)
    # step 1: Build cost Matrix
    if num_dets > 0 and num_trajs > 0:
        cost_matrix = np.full(
            (num_dets, num_trajs),
            1e6,
            dtype=np.float32
        )

        for det_pos, i in enumerate(det_indices):
            class_id = class_ids[i]
            box = boxes[i]
            det_feat = dets_feats[i]
            for traj_pos, traj in enumerate(trajectory):
                # check class id
                if traj["class_id"] != class_id:
                    continue

                dt = ts - traj["last_ts"]

                if dt <= 0:
                    continue

                iou = compute_iou(traj["last_box"], box, traj["vx"], traj["vy"], dt)
                if iou < 0.1:
                    continue

                cost_iou = 1.0 - iou

                traj_feat_bank = traj["feat_bank"]
                if len(traj_feat_bank) == 0:
                    cost_reid = 0.0
                else:
                    max_sim = 0.0


                    for feat in traj_feat_bank:
                        # compute similarity
                        sim = np.dot(det_feat, feat)
                        if sim > max_sim:
                            max_sim = sim

                    if max_sim < 0.1:
                        continue

                    cost_reid = 1.0 - max_sim

                total_cost = 0.7 * cost_iou + 0.3 * cost_reid

                # cost = 1 - iou
                # lower is better
                cost_matrix[det_pos, traj_pos] = total_cost

        # Step 2: hungarian algorithm to solve the assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Step 3: filter valid assignments
        for det_pos, traj_pos in zip(row_indices, col_indices):
            if cost_matrix[det_pos, traj_pos] >= 1e5:
                continue

            det_idx = det_indices[det_pos]
            traj = trajectory[traj_pos]

            # detection from prediction
            matched_dets.add(det_idx)

            # matched trajectory unique id
            matched_indices.add(traj["id"])

            # update trajectory with matched detections
            dt = ts - traj["last_ts"]
            vx = (boxes[det_idx][0] - traj["last_box"][0]) / dt
            vy = (boxes[det_idx][1] - traj["last_box"][1])
            traj["score"] = scores[det_idx]
            traj["last_box"] = boxes[det_idx]
            traj["last_ts"] = ts
            traj["vx"] = vx
            traj["vy"] = vy
            traj["lost_cnt"] = 0        # if appear in this frame, reset lost count
            traj["history"].append((ts, boxes[det_idx]))
            traj["feat_bank"].append(det_feat)

    # step 4: increment lost count for unmatched trajectories
    for traj in trajectory:
        if traj["id"] not in matched_indices:
            traj["lost_cnt"] += 1

    # step 5: create new trajectory elements for unmatched detections
    for det_idx in det_indices:
        if det_idx in matched_dets:
            continue

        global next_track_id

        traj_element = {
            "id": next_track_id,
            "class_id": class_ids[det_idx],
            "score": scores[det_idx],
            "last_box": boxes[det_idx],
            "last_ts": ts,
            "vx": 0.0,
            "vy": 0.0,
            "feat_bank": [dets_feats[det_idx]],
            "lost_cnt": 0,
            "history": []
        }
        next_track_id+=1

        trajectory.append(traj_element)



    # step 6: remove trajectories that have been lost for too long
    trajectory[:] = [
        traj for traj in trajectory
        if traj["lost_cnt"] <= MAX_LOST_CNT
    ]

if __name__ == "__main__":
    # =========================
    # Warm-up
    # =========================

    dummy = np.random.randint(0,255, (640, 640, 3), dtype=np.uint8)
    dummy_tensor, _, _, _ = preprocess(dummy)

    print("Warmup...")
    for _ in range(10):
        _ = session_det.run([output_info.name], {input_info.name: dummy_tensor})
    print("Warmup done.")


    # =========================
    # Producer Thread
    # =========================
    producer_thread = threading.Thread(target=frame_producer)
    producer_thread.start()

    frame_processed = 0
    total_start = time.perf_counter()


    # =========================
    # streaming inference
    # =========================
    try:
        print("Start streaming inference...")
        while not stop_event.is_set():
            try:
                frame, ts = frame_queue.get(timeout=1)
            except queue.Empty:
                continue


            t0 = time.perf_counter()

            input_tensor, scale, left, top = preprocess(frame)

            outputs = session_det.run(
                [output_info.name],
                {input_info.name: input_tensor}
            )
            t1 = time.perf_counter()

            inference_time = (t1 - t0) * 1000
            frame_processed += 1

            predictions = outputs[0][0]  # shape: (84, 8400)
            predictions = np.transpose(predictions, (1, 0)) # [84, 8400] -> [8400, 84]


            boxes, scores, class_ids, indices = post_process(predictions, scale, left, top, frame)
            # feature cache detection features for each box
            feature = extract_reid_feature(frame, boxes, session_reid)
            track_detection(trajectory, boxes, scores, class_ids, indices, feature, ts)
            # for traj in trajectory:
            #     print(f"ID: {traj['id']}, Class: {class_names[traj['class_id']]}, Score: {traj['score']:.2f}, Box: {traj['last_box']}, Lost Count: {traj['lost_cnt']}")
            draw_trajectory(frame, trajectory)

            # draw_results(frame, boxes, scores, class_ids, indices, trajectory)



    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        stop_event.set()
        producer_thread.join()

        total_end = time.perf_counter()
        total_time = total_end - total_start

        print(f"Total frames processed: {frame_processed}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average FPS: {frame_processed / total_time:.2f}")
        print(f"Dropped frames: {drop_cnt}")
