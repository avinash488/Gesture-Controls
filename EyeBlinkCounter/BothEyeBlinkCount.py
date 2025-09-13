import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
from _collections import deque

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 360)

detector = FaceMeshDetector(maxFaces=1)

plotY = LivePlot(640,360,[20,50], invert=True)

idList = [22,23,24,26,110,157,158,159,160,161,130,243,
          362,385,386,387,263,373,374,380,381,382,384]
ratioList = []
blinkCount = 0
counter = 0
color = (255, 0, 255)

while True:
    success, img = cap.read()
    img = cv2.resize(img, (640, 360))
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        for id in idList:
            cv2.circle(img, face[id],5,(255,0,255),cv2.FILLED)
        leftUp = face[159]
        leftDown = face[23]
        leftLeftEdge = face[130]
        leftRightEdge = face[243]
        rightUp = face[386]
        rightDown = face[384]
        rightLeftEdge = face[362]
        rightRightEdge = face[263]
        gap_left_Up_Down,_ = detector.findDistance(leftUp,leftDown)
        gap_left_Left_Right,_ = detector.findDistance(leftLeftEdge,leftRightEdge)
        #cv2.line(img,leftUp,leftDown,(0,200,255),2)
        #cv2.line(img,leftLeftEdge,leftRightEdge,(0,200,200),2)
        gap_right_Up_Down,_ = detector.findDistance(rightUp,rightDown)
        gap_right_Left_Right,_ = detector.findDistance(rightLeftEdge,rightRightEdge)

        rratio = int(gap_right_Up_Down / gap_right_Left_Right)*100
        lratio = int((gap_left_Up_Down/gap_left_Left_Right)*100)
        pratio = int(lratio + rratio)
        ratioList.append(pratio)
        if len(ratioList) > 4:
            ratioList.pop(0)
        ratioAverage = sum(ratioList)/len(ratioList)

        if ratioAverage < 35 and counter == 0:
            blinkCount += 1
            color = (0, 200, 0)
            counter = 1
        if counter != 0:
            counter += 1
            if counter > 10:
                counter = 0
                color = (255, 0, 255)
        cvzone.putTextRect(img, f'BlinkCount: {blinkCount}', (50,100), colorR=color)

        imgPlot = plotY.update(ratioAverage, color)
        cv2.imshow("LivePlot", imgPlot)
        imgStack = cvzone.stackImages([img,imgPlot],2,1)
    else:
        cv2.resizeWindow('LivePlot',640, 360)
        imgStack = cvzone.stackImages([img,img],2,1)
    cv2.imshow('frame', imgStack)
    cv2.waitKey(25)
