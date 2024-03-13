import math

import cv2
import mediapipe as mp


class poseDetector():
    def __init__(self, mode=False, upbody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upbody = upbody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        # self.pose = self.mpPose.Pose(self.mode, self.upbody, self.smooth, self.detectionCon, self.trackCon)
        self.pose = self.mpPose.Pose()
        self.mpPose.Pose()

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        # print(results.pose_landmarks)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (0, 0, 255), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
        # print(angle)

        if draw:
            cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x3, y3), 10, (255, 0, 0), cv2.FILLED)

            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.line(img, (x2, y2), (x3, y3), (255, 0, 0), 3)

        return angle


def main():
    cap = cv2.VideoCapture(1)
    detector = poseDetector()
    while True:
        success, img = cap.read()
        detector.findPose(img)

        cv2.imshow('img', img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
