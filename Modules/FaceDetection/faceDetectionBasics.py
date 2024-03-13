import cv2
import mediapipe as mp

cap = cv2.VideoCapture(1)


mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection()
mpDraw = mp.solutions.drawing_utils



while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    # print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # print(id, detection)
            # mpDraw.draw_detection(img, detection)
            print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            # print(bbox)
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, f'{round(detection.score[0]*100)}%', (bbox[0]-20, bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)



    cv2.imshow('img', img)
    cv2.waitKey(1)