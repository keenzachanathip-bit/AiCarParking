from ultralytics import YOLO
import cv2
import json
import numpy as np
import subprocess

# =========================
# LOAD MODEL
# =========================

model = YOLO("model/yolov8s.pt")

# =========================
# YOUTUBE LIVE STREAM
# =========================

def get_stream_url(youtube_url):
    result = subprocess.run(
        ["yt-dlp", "--js-runtimes", "node",
         "--remote-components", "ejs:github",
         "-f", "93", "-g", youtube_url],
        capture_output=True, text=True
    )
    return result.stdout.strip()

YOUTUBE_URL = "https://www.youtube.com/watch?v=EPKWu223XEg"

print("Getting stream URL...")
stream_url = get_stream_url(YOUTUBE_URL)
print("Stream ready!")

cap = cv2.VideoCapture(stream_url)

# =========================
# LOAD SLOT DATA
# =========================

with open("slots/slots.json") as f:
    slots = json.load(f)

# =========================
# MATCH RESOLUTION
# =========================

ref = cv2.imread("tools/frame_parking.png")
ref_h, ref_w = ref.shape[:2]

# =========================
# INTERSECTION FUNCTION
# =========================

def intersection_ratio(box, polygon, frame_shape):

    x1,y1,x2,y2 = box

    car_poly = np.array([
        [x1,y1],
        [x2,y1],
        [x2,y2],
        [x1,y2]
    ], np.int32)

    car_mask = np.zeros((frame_shape[0],frame_shape[1]),dtype=np.uint8)
    slot_mask = np.zeros_like(car_mask)

    cv2.fillPoly(car_mask,[car_poly],255)
    cv2.fillPoly(slot_mask,[polygon],255)

    inter = cv2.bitwise_and(car_mask,slot_mask)

    inter_area = cv2.countNonZero(inter)
    car_area = cv2.countNonZero(car_mask)

    if car_area == 0:
        return 0

    return inter_area / car_area


# =========================
# MAIN LOOP
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Stream lost, reconnecting...")
        stream_url = get_stream_url(YOUTUBE_URL)
        cap = cv2.VideoCapture(stream_url)
        continue

    frame = cv2.resize(frame, (ref_w, ref_h))

    results = model(frame, conf=0.4)

    car_boxes = []

    for r in results:

        boxes = r.boxes.xyxy
        classes = r.boxes.cls

        for box, cls in zip(boxes, classes):

            cls = int(cls)

            if cls not in [2,5,7]:
                continue

            x1,y1,x2,y2 = map(int,box)

            car_boxes.append((x1,y1,x2,y2))

            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)

    occupied_count = 0

    for slot in slots:

        pts = np.array(slot["points"],np.int32)

        occupied = False

        for box in car_boxes:

            overlap = intersection_ratio(box, pts, frame.shape)

            if overlap > 0.25:
                occupied = True
                break

        if occupied:
            color = (0,0,255)
            occupied_count += 1
        else:
            color = (0,255,0)

        cv2.polylines(frame,[pts],True,color,2)

    total_slots = len(slots)
    free_slots = total_slots - occupied_count

    cv2.putText(frame,
                f"Free: {free_slots}",
                (30,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    cv2.putText(frame,
                f"Occupied: {occupied_count}",
                (30,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                2)

    cv2.imshow("AI Smart Parking (YouTube Live)", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()