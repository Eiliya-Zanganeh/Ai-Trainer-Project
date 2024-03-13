import cv2
import mediapipe as mp
import time


class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHand = mp.solutions.hands
        self.hands = self.mpHand.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHand.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        hands = []
        if self.results.multi_hand_landmarks:
            for item in list(zip(self.results.multi_handedness, self.results.multi_hand_landmarks)):
                # hand = []
                fingers = []
                for id, lm in enumerate(item[1].landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    fingers.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                hands.append([item[0].classification[0].label, fingers])
        return hands

    def fingersUp(self, img, reverse=False, hand_custome:list=[]):
        arrow = ['Left', 'Right']
        if reverse:
            arrow.reverse()
        ids = [4, 8, 12, 16, 20]
        fingers = [['Left', [0, 0, 0, 0, 0]], ['Right', [0, 0, 0, 0, 0]]]
        count = 0
        if hand_custome == []:
            hands = self.findPosition(img)
        else:
            hands = [hand_custome]
        if len(hands) != 0:
            for hand in hands:
                if hand[0] == 'Left':
                    for num in ids:
                        if num == 4:
                            if hand[1][num][1] > hand[1][num - 1][1]:
                                count += 1
                                fingers[arrow.index('Left')][1][ids.index(num)] = 1
                        else:
                            if hand[1][num][2] < hand[1][num - 2][2]:
                                count += 1
                                fingers[arrow.index('Left')][1][ids.index(num)] = 1
                if hand[0] == 'Right':
                    for num in ids:
                        if num == 4:
                            if hand[1][num][1] < hand[1][num - 1][1]:
                                count += 1
                                fingers[arrow.index('Right')][1][ids.index(num)] = 1
                        else:
                            if hand[1][num][2] < hand[1][num - 2][2]:
                                count += 1
                                fingers[arrow.index('Right')][1][ids.index(num)] = 1
        if fingers[0][1] == [0, 0, 0, 0, 0]:
            fingers.remove(fingers[0])
        elif fingers[1][1] == [0, 0, 0, 0, 0]:
            fingers.remove(fingers[1])
        return count, fingers


def main():
    cap = cv2.VideoCapture(1)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        # hands = detector.findPosition(img)
        # if len(hands) != 0:
        #     for hand in hands:
        #         for arrow, landmark in hand:
        #             print(arrow, landmark)
        detector.fingersUp(img)
        cv2.imshow('img', img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
