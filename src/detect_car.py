from ultralytics import YOLO
import cv2

model = YOLO("model/yolov8s.pt")

cap = cv2.VideoCapture("video/parking_test.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated = results[0].plot()

    cv2.imshow("Parking Detection", annotated)

    if cv2.waitKey(1) == 27:
        break