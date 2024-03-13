import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1)
mpHand = mp.solutions.hands
hands = mpHand.Hands()
mpDraw = mp.solutions.drawing_utils


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS)
    # print(time.time())
    cv2.imshow('img', img)
    cv2.waitKey(1)