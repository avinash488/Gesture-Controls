import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import numpy as np

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)

textList = ["Welcome to Avi's Code","It's a code on Computer Vision",
            "The code changes text","size based on your","distance from the screen"]
sen = 15
while True:
    success, img = cap.read()
    imgText = np.zeros_like(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]
        # cv2.line(img, pointLeft, pointRight, (0, 200, 0), 2)
        # cv2.circle(img, pointRight, 5, (255, 0, 0), cv2.FILLED)
        # cv2.circle(img, pointLeft, 5, (255, 0, 0), cv2.FILLED)

        # Finding the Focal Length
        w, _ = detector.findDistance(pointLeft, pointRight)
        W = 6.3


        # Finding Depth

        f = 570
        d = (W * f) / w
        print(d)

        cvzone.putTextRect(img, f'Depth: {int(d)}cm', (face[10][0] - 75, face[10][1] - 50),scale = 2)
        for i,text in enumerate(textList):
            singleHeight = 20 + int((int(d/sen)*sen)/4)
            scaleL = 0.4 + (int(d/sen)*sen)/75
            cv2.putText(imgText, text, (50,50+(i*singleHeight)), cv2.FONT_HERSHEY_SIMPLEX, scaleL, (255,255,255), 2)

    imgStacked = cvzone.stackImages([img, imgText], 2, 1)
    cv2.imshow("Face Detection", imgStacked)
    cv2.waitKey(1)