import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)



while True:
    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]
        #cv2.line(img, pointLeft, pointRight, (0, 200, 0), 2)
        #cv2.circle(img, pointRight, 5, (255, 0, 0), cv2.FILLED)
        #cv2.circle(img, pointLeft, 5, (255, 0, 0), cv2.FILLED)

        # Finding the Focal Length
        w,_ = detector.findDistance(pointLeft, pointRight)
        W = 6.3
        #d = 50
        #f = (w*d)/W
        #print(f)

        # Finding Depth

        f = 570
        d = (W*f)/w
        print(d)

        cvzone.putTextRect(img,f'Depth: {int(d)}cm',(face[10][0] - 75,face[10][1] - 50))

    cv2.imshow("Face Detection", img)
    cv2.waitKey(1)