import cv2
import numpy as np
from collections import deque

dq = deque(maxlen=3)
buffersize = 16
pts = deque(maxlen=buffersize)

redlower = (160, 100, 100)
redupper = (179, 255, 255)

cap = cv2.VideoCapture(1)
cap.set(3, 1920)
cap.set(4, 1080)

while True:
    success, imgoriginal = cap.read()
    if success:
        blurred = cv2.GaussianBlur(imgoriginal, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, redlower, redupper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        center = None

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(c)
            ((x, y), (width, height), rotation) = rect

            s = "x:{}, y:{}, width:{}, height:{}, rotation:{}".format(np.round(x), np.round(y), np.round(width), np.round(height), rotation)
            print(s)

            box = cv2.boxPoints(rect)
            box = np.int0(box)

            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            cv2.drawContours(imgoriginal, [box], 0, (0, 255, 255), 2)
            cv2.circle(imgoriginal, center, 5, (255, 0, 0), -1)

            pts.appendleft(center)

            for i in range(1, len(pts)):
                if pts[i - 1] is None or pts[i] is None:
                    continue

                cv2.line(imgoriginal, pts[i - 1], pts[i], (0, 255, 255), 3)

            cv2.imshow("Original", imgoriginal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
