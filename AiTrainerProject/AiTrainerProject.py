import math
# from builtins import str

import cv2
from Modules.PoseEstimation import poseEstimationModule as pe

# cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture('video.mp4')
detector = pe.poseDetector()
count = [True, 0]

while True:
    success, img = cap.read()
    img = detector.findPose(img, draw=False)
    lmList = detector.getPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[23][1:]
        x2, y2 = lmList[24][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        # cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

        x1, y1 = lmList[27][1:]
        x2, y2 = cx, cy
        x3, y3 = lmList[28][1:]
        while True:
            angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
            if angle < 0:
                angle += 360

            if angle > 200:
                x1, y1 = lmList[28][1:]
                x2, y2 = cx, cy
                x3, y3 = lmList[27][1:]
            else:
                break
        # print(angle)
        cv2.putText(img, f'{round(angle)} ^', (50, 300), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(img, f'{count[1]}', (50, 350), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
        if round(angle) > 130 and count[0]:
            count[1] += 1
            count[0] = False
        if round(angle) < 10 and not count[0]:
            count[0] = True




        cv2.circle(img, (x1, y1), 5, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 5, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x3, y3), 5, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.line(img, (x2, y2), (x3, y3), (255, 0, 0), 3)




    cv2.imshow('img', img)
    cv2.waitKey(20)