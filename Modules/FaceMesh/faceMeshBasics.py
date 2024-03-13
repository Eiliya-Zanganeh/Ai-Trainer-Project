import cv2
import mediapipe as mp

cap = cv2.VideoCapture(1)

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
fashMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = fashMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLam in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLam, mpFaceMesh.FACEMESH_CONTOURS, landmark_drawing_spec=drawSpec, connection_drawing_spec=drawSpec)

            for id, lm in enumerate(faceLam.landmark):
                ih, iw, ic = img.shape

                x, y = int(lm.x * iw), int(lm.y * ih)
                print(id, x, y)

    cv2.imshow('img', img)
    cv2.waitKey(1)