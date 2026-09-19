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
         "-f", "best", "-g", youtube_url],
        capture_output=True, text=True
    )
    return result.stdout.strip()

YOUTUBE_URL = "https://www.youtube.com/watch?v=EPKWu223XEg"

print("Getting stream URL...")
stream_url = get_stream_url(YOUTUBE_URL)
print("Stream ready!")

cap = cv2.VideoCapture(stream_url)

# =========================
# RESOLUTION
# =========================

PROC_W, PROC_H = 1920, 1080   # resolution ที่ตีกรอบ slot ไว้
DISP_W, DISP_H = 1280, 720    # resolution ที่จะ display

scale_x = DISP_W / PROC_W
scale_y = DISP_H / PROC_H

# =========================
# LOAD + PRE-SCALE SLOT DATA
# =========================

with open("slots/slots.json") as f:
    slots_raw = json.load(f)

slots = []
for slot in slots_raw:
    scaled_pts = [[int(x * scale_x), int(y * scale_y)] for x, y in slot["points"]]
    slots.append({"points": scaled_pts})

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

    frame = cv2.resize(frame, (DISP_W, DISP_H))

    results = model(frame, conf=0.4, verbose=False)

    car_centers = []

    for r in results:

        boxes = r.boxes.xyxy
        classes = r.boxes.cls

        for box, cls in zip(boxes, classes):

            cls = int(cls)

            if cls not in [2, 5, 7]:  # car, bus, truck
                continue

            x1, y1, x2, y2 = map(int, box)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            car_centers.append((cx, cy))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

    occupied_count = 0

    for slot in slots:

        pts = np.array(slot["points"], np.int32)

        occupied = False

        for cx, cy in car_centers:

            if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0:
                occupied = True
                break

        if occupied:
            color = (0, 0, 255)
            occupied_count += 1
        else:
            color = (0, 255, 0)

        cv2.polylines(frame, [pts], True, color, 2)

    total_slots = len(slots)
    free_slots = total_slots - occupied_count

    cv2.putText(frame, f"Free: {free_slots}",
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Occupied: {occupied_count}",
                (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("AI Smart Parking", frame)

    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()