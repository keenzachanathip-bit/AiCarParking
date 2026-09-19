import cv2

for i in range(3):

    cap = cv2.VideoCapture(i)

    if not cap.isOpened():
        continue

    print("Testing Camera", i)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        cv2.imshow(f"Camera {i}", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()