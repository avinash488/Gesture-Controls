import cv2
import mediapipe as mp
import time
import math
import numpy as np
from pynput.keyboard import Controller
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

# -------------------- Hand Detector --------------------
class HandDetector:
    def __init__(self, detectionCon=0.8, maxHands=1):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=maxHands,
            min_detection_confidence=detectionCon,
            min_tracking_confidence=0.5)
        self.results = None
        self.lmList = []

    def findHands(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results and self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            h, w, _ = img.shape
            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
        return self.lmList

    def findDistance(self, p1, p2, img, draw=True):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        length = math.hypot(x2 - x1, y2 - y1)
        return length, [x1, y1, x2, y2, (x1 + x2) // 2, (y1 + y2) // 2]


# -------------------- Futuristic Button --------------------
class Button:
    def __init__(self, pos, text, size=(85, 85)):
        self.pos = pos
        self.size = size
        self.text = text

    def draw(self, img, state="idle"):
        x, y = self.pos
        w, h = self.size
        H, W, _ = img.shape

        # Clip coordinates to image boundaries
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)

        overlay = img.copy()

        # Sci-fi hologram colors
        if state == "idle":
            color, alpha = (107, 142, 35), 0.25
        elif state == "hover":
            color, alpha = (0, 255, 255), 0.35
        elif state == "click":
            color, alpha = (255, 0, 255), 0.5
        else:
            color, alpha = (200, 200, 200), 0.2

        roi = img[y1:y2, x1:x2]
        if roi.size > 0:
            blur = cv2.GaussianBlur(roi, (35, 35), 30)
            img[y1:y2, x1:x2] = blur

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, cv2.FILLED)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        cv2.rectangle(img, (x1+5, y1+5), (x2-5, y2-5), color, 2)

        # Text rendering
        text_img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(text_img, self.text, (15, int(h/1.6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 4)
        cv2.putText(text_img, self.text, (15, int(h/1.6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
        flipped = cv2.flip(text_img, 1)

        roi = img[y1:y2, x1:x2]
        if roi.shape[:2] == flipped.shape[:2]:
            img[y1:y2, x1:x2] = cv2.addWeighted(roi, 1, flipped, 1, 0)

    def isHovered(self, x, y):
        bx, by = self.pos
        bw, bh = self.size
        return bx < x < bx + bw and by < y < by + bh


# -------------------- Main --------------------
def main():
    cap = cv2.VideoCapture(0) #R
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = HandDetector(detectionCon=0.8)
    keyboard = Controller()

    # Blink Detector
    faceDetector = FaceMeshDetector(maxFaces=1)
    blinkThreshold = 0.25  # 👀 EAR threshold
    blinkCooldown = 5      # frames before reset

    # Keyboard layout
    keys = [["Q","W","E","R","T","Y","U","I","O","P"],
            ["A","S","D","F","G","H","J","K","L",";"],
            ["Z","X","C","V","B","N","M",",",".","/"]]

    buttonList = []
    for i in range(len(keys)):
        for j, key in enumerate(keys[i]):
            buttonList.append(Button((100 * j + 50, 100 * i + 50), key))

    # Special row
    buttonList.append(Button((150, 400), "Space", size=(400, 85)))
    buttonList.append(Button((600, 400), "Backspace", size=(250, 85)))
    buttonList.append(Button((900, 400), "Enter", size=(250, 85)))

    finalText = ""
    lastClick = 0
    trail = []

    # Blink detection vars
    blinkCount = 0
    counter = 0
    lastBlinkTime = 0
    useBlinkClick = False   # start in pinch mode

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)

        # Hand detection
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        # Face detection for blink
        img, faces = faceDetector.findFaceMesh(img, draw=False)
        longBlink = False

        if faces:
            face = faces[0]
            leftUp, leftDown, leftLeft, leftRight = face[159], face[23], face[130], face[243]
            rightUp, rightDown, rightLeft, rightRight = face[386], face[374], face[362], face[263]

            leftUD, _ = faceDetector.findDistance(leftUp, leftDown)
            leftLR, _ = faceDetector.findDistance(leftLeft, leftRight)
            rightUD, _ = faceDetector.findDistance(rightUp, rightDown)
            rightLR, _ = faceDetector.findDistance(rightLeft, rightRight)

            lratio = leftUD / leftLR
            rratio = rightUD / rightLR
            ear = (lratio + rratio) / 2

            if ear < blinkThreshold and counter == 0:
                blinkCount += 1
                counter = 1
                cv2.circle(img, (leftUp[0], leftUp[1]), 15, (0,0,255), -1)  # red = blink
                if time.time() - lastBlinkTime > 0.9:  # double blink window
                    longBlink = True
                lastBlinkTime = time.time()

            if counter != 0:
                counter += 1
                if counter > blinkCooldown:
                    counter = 0

        # Draw buttons
        for button in buttonList:
            button.draw(img, "idle")

        if lmList:
            x, y = lmList[8][1:]
            mx, my = lmList[12][1:]

            trail.append((x, y, time.time()))
            for (tx, ty, t) in trail:
                fade = max(0, 1 - (time.time() - t) / 0.6)
                if fade > 0:
                    cv2.circle(img, (tx, ty), 15, (0, int(255*fade), int(255*fade)), -1)
            trail = [(tx, ty, t) for (tx, ty, t) in trail if time.time() - t < 0.6]

            cv2.circle(img, (x, y), 12, (255, 0, 255), cv2.FILLED)

            for button in buttonList:
                if button.isHovered(x, y):
                    button.draw(img, "hover")

                    # Pinch click
                    if not useBlinkClick:
                        length, _ = detector.findDistance(8, 12, img, draw=False)
                        if length < 40 and time.time() - lastClick > 0.5:
                            key = button.text
                            if key == "Space":
                                finalText += " "
                            elif key == "Backspace":
                                finalText = finalText[:-1]
                            elif key == "Enter":
                                finalText += "\n"
                            else:
                                finalText += key
                            button.draw(img, "click")
                            lastClick = time.time()

                    # Double blink click
                    elif useBlinkClick and longBlink and time.time() - lastClick > 0.8:
                        key = button.text
                        if key == "Space":
                            finalText += " "
                        elif key == "Backspace":
                            finalText = finalText[:-1]
                        elif key == "Enter":
                            finalText += "\n"
                        else:
                            finalText += key
                        # Green flash to confirm double blink click
                        cv2.rectangle(img, (button.pos[0], button.pos[1]),
                                      (button.pos[0]+button.size[0], button.pos[1]+button.size[1]),
                                      (0,255,0), 4)
                        lastClick = time.time()

        # Output text box
        overlay = img.copy()
        cv2.rectangle(overlay, (50, 500), (1200, 600), (107, 142, 35), cv2.FILLED)
        cv2.addWeighted(overlay, 0.12, img, 0.88, 0, img)
        cv2.rectangle(img, (50, 500), (1200, 600), (0, 255, 255), 2)
        cv2.putText(img, finalText[-30:], (60, 570),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)

        # Show mode indicator
        modeText = "Mode: Double Blink" if useBlinkClick else "Mode: Pinch"
        cv2.putText(img, modeText, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)

        cv2.imshow("Sci-Fi Hologram Keyboard", img)
        keyPress = cv2.waitKey(1) & 0xFF
        if keyPress == ord('q'):
            break
        elif keyPress == ord('*'):  # Toggle mode with *
            useBlinkClick = not useBlinkClick
            time.sleep(0.3)  # debounce

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
