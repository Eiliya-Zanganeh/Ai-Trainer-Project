import cv2
import mediapipe as mp



class FaceMeshDetector:
    def __init__(self, static_image_mode=False, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.fashMesh = self.mpFaceMesh.FaceMesh(self.static_image_mode, self.max_num_faces, self.refine_landmarks, self.min_detection_confidence, self.min_tracking_confidence)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.fashMesh.process(imgRGB)
        faces = []
        if results.multi_face_landmarks:
            for faceLam in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLam, self.mpFaceMesh.FACEMESH_CONTOURS, landmark_drawing_spec=self.drawSpec, connection_drawing_spec=self.drawSpec)
                    face = []
                    for id, lm in enumerate(faceLam.landmark):
                        ih, iw, ic = img.shape

                        x, y = int(lm.x * iw), int(lm.y * ih)
                        # print(id, x, y)
                        face.append([id, x, y])
                    faces.append(face)
        return img, faces












def main():
    cap = cv2.VideoCapture(1)
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        img = detector.findFaceMesh(img)
        print(img[1])
        cv2.imshow('img', img[0])
        cv2.waitKey(1)


if __name__ == '__main__':
    main()