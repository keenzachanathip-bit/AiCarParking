from ultralytics import YOLO
import cv2
import json
import numpy as np

# load YOLO model
model = YOLO("model/yolov8s.pt")

# open video
cap = cv2.VideoCapture("video/parking_test.mp4")

# load slot data
with open("slots/slots.json") as f:
    slots = json.load(f)

while True:

    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, conf=0.4)

    car_centers = []

    for r in results:

        boxes = r.boxes.xyxy
        classes = r.boxes.cls

        for box, cls in zip(boxes, classes):

            cls = int(cls)

            # vehicle classes
            if cls not in [2,5,7]:
                continue

            x1,y1,x2,y2 = map(int,box)

            cx = int((x1+x2)/2)
            cy = int((y1+y2)/2)

            car_centers.append((cx,cy))

            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.circle(frame,(cx,cy),4,(0,255,255),-1)

    occupied_count = 0

    for slot in slots:

        pts = np.array(slot["points"],np.int32)

        occupied = False

        for cx,cy in car_centers:

            inside = cv2.pointPolygonTest(pts,(cx,cy),False)

            if inside >= 0:
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
    
    display = cv2.resize(frame,(1280,720))
    cv2.imshow("AI Smart Parking",display)
    

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()