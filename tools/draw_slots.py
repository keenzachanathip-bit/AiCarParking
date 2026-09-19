import cv2
import json
import os
import numpy as np

base_path = os.path.dirname(__file__)
image_path = os.path.join(base_path, "frame_parking02.png")

image = cv2.imread(image_path)

if image is None:
    print("Image not found:", image_path)
    exit()

slots = []
points = []

def draw_view():

    view = image.copy()

    # draw saved slots
    for slot in slots:

        pts = np.array(slot["points"], np.int32)

        cv2.polylines(view, [pts], True, (0,255,0), 2)

    # draw current points
    for px,py in points:

        cv2.circle(view, (px,py), 4, (0,255,0), -1)

    return view


def mouse(event,x,y,flags,param):

    global points

    if event == cv2.EVENT_LBUTTONDOWN:

        points.append((x,y))

        print("Point:",x,y)

        if len(points)==4:

            slot = {
                "id":len(slots)+1,
                "points":points.copy()
            }

            slots.append(slot)

            points.clear()


cv2.namedWindow("Draw Slots", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Draw Slots", mouse)

print("Controls:")
print("Left click = slot point")
print("4 clicks = create slot")
print("U = undo last slot")
print("ESC = save")

while True:

    view = draw_view()

    cv2.imshow("Draw Slots", view)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

    if key == ord("u"):

        if len(slots) > 0:
            slots.pop()

slots_path = os.path.join(base_path,"..","slots","slots.json")

with open(slots_path,"w") as f:
    json.dump(slots,f,indent=4)

print("Slots saved:",slots_path)

cv2.destroyAllWindows()