from typing import Deque
import cv2
import numpy as np
import HandTrakingModule as htm
import time
from collections import deque


# default called trackbar function
def setValues(x):
    print("")


# Giving different arrays to handle colour points of different colour
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

# These indexes will be used to mark the points in particular arrays of specific colour
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

# The kernel to be used for dilation purpose
kernel = np.ones((5, 5), np.uint8)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

# Here is code for Canvas setup
paintWindow = np.zeros((471, 636, 3)) + 255
paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), colors[0], -1)
paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), colors[1], -1)
paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), colors[2], -1)
paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), colors[3], -1)

cv2.putText(paintWindow, "CLEAR", (49, 33),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (298, 33),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (520, 33),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2, cv2.LINE_AA)
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

#########################################################
cam_width, cam_height = 640, 480
#########################################################
cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)
pTime = 0

detector = htm.handDetector(min_dectection_confidence=0.9)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img, draw=False)
    lmList = detector.findPosition(img, draw=False)
    print(lmList)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    # Adding the colour buttons to the live frame for colour access
    img = cv2.rectangle(img, (40, 1), (140, 65), (122, 122, 122), -1)
    img = cv2.rectangle(img, (160, 1), (255, 65), colors[0], -1)
    img = cv2.rectangle(img, (275, 1), (370, 65), colors[1], -1)
    img = cv2.rectangle(img, (390, 1), (485, 65), colors[2], -1)
    img = cv2.rectangle(img, (505, 1), (600, 65), colors[3], -1)
    cv2.putText(img, "CLEAR ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (150, 150, 150), 2, cv2.LINE_AA)

    cv2.putText(img, f'FPS:{int(fps)}', (40, 110),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    if len(lmList) != 0:

        x1, y1 = lmList[8][1], lmList[8][2]
        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        if lmList[8][1] < lmList[7][1]:
            if y1 <= 65:
                if 40 <= x1 <= 140:  # Clear Button
                    bpoints = [deque(maxlen=512)]
                    gpoints = [deque(maxlen=512)]
                    rpoints = [deque(maxlen=512)]
                    ypoints = [deque(maxlen=512)]

                    blue_index = 0
                    green_index = 0
                    red_index = 0
                    yellow_index = 0
                    paintWindow[67:, :, :] = 255
                elif 160 <= x1 <= 255:
                    print("Blue")
                    colorIndex = 0  # Blue
                elif 275 <= x1 <= 370:
                    print("Green")
                    colorIndex = 1  # Green
                elif 390 <= x1 <= 485:
                    print("Red")
                    colorIndex = 2  # Red
                elif 505 <= x1 <= 600:
                    print("Yellow")
                    colorIndex = 3  # Yellow
            else:
                if colorIndex == 0:
                    bpoints[blue_index].appendleft((x1, y1))
                elif colorIndex == 1:
                    gpoints[green_index].appendleft((x1, y1))
                elif colorIndex == 2:
                    rpoints[red_index].appendleft((x1, y1))
                elif colorIndex == 3:
                    ypoints[yellow_index].appendleft((x1, y1))
        # Append the next deques when nothing is detected to avois messing up
        else:
            bpoints.append(deque(maxlen=512))
            blue_index += 1
            gpoints.append(deque(maxlen=512))
            green_index += 1
            rpoints.append(deque(maxlen=512))
            red_index += 1
            ypoints.append(deque(maxlen=512))
            yellow_index += 1

    points = [bpoints, gpoints, rpoints, ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(img, points[i][j][k - 1],
                         points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j]
                         [k - 1], points[i][j][k], colors[i], 2)

    cv2.imshow("Img", img)
    cv2.imshow("Paint", paintWindow)
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and all resources
cap.release()
cv2.destroyAllWindows()
