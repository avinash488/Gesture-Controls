import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 360)

detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640, 360, [20, 50], invert=True)

# landmark indices for both eyes
idList = [159, 23, 130, 243,   # left eye (top, bottom, left, right)
          386, 384, 362, 263]  # right eye (top, bottom, left, right)

ratioList = []
blinkCount = 0
counter = 0
color = (255, 0, 255)

while True:
    success, img = cap.read()
    img = cv2.resize(img, (640, 360))
    img = cv2.flip(img, 1)
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]

        # Landmarks for left eye
        leftUp = face[159]
        leftDown = face[23]
        leftLeftEdge = face[130]
        leftRightEdge = face[243]

        # Landmarks for right eye
        rightUp = face[386]
        rightDown = face[374]
        rightLeftEdge = face[362]
        rightRightEdge = face[263]

        # Distances for left eye
        gap_left_Up_Down, _ = detector.findDistance(leftUp, leftDown)
        gap_left_Left_Right, _ = detector.findDistance(leftLeftEdge, leftRightEdge)
        cv2.line(img,leftUp,leftDown,(0,200,255),2)
        cv2.line(img,leftLeftEdge,leftRightEdge,(0,200,200),2)
        # Distances for right eye
        gap_right_Up_Down, _ = detector.findDistance(rightUp, rightDown)
        gap_right_Left_Right, _ = detector.findDistance(rightLeftEdge, rightRightEdge)
        cv2.line(img,rightUp,rightDown,(0,200,255),2)
        cv2.line(img,rightLeftEdge,rightRightEdge,(0,200,200),2)
        # Normalized EAR (rotation-invariant)
        lratio = int((gap_left_Up_Down / gap_left_Left_Right)*100)
        rratio = int((gap_right_Up_Down / gap_right_Left_Right)*100)

        # Average EAR of both eyes
        pratio = int((lratio + rratio) / 2.0)

        # Smooth ratio over last few frames
        ratioList.append(pratio)
        if len(ratioList) > 4:
            ratioList.pop(0)
        ratioAverage = sum(ratioList) / len(ratioList)

        # Blink detection logic
        if ratioAverage < 28 and counter == 0:   # adjust threshold if needed
            blinkCount += 1
            color = (0, 200, 0)
            counter = 1
        if counter != 0:
            counter += 1
            if counter > 10:   # debounce
                counter = 0
                color = (255, 0, 255)

        # Show blink counter only
        cvzone.putTextRect(img, f'BlinkCount: {blinkCount}', (20, 50), colorR=color)

        # Live plot (optional)
        imgPlot = plotY.update(ratioAverage, color)
        imgStack = cvzone.stackImages([img, imgPlot], 2, 1)

    else:
        imgStack = cvzone.stackImages([img, img], 2, 1)

    cv2.imshow('frame', imgStack)
    cv2.waitKey(25)   # press Esc to exit


