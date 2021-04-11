import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, max_num_hand=2, min_dectection_confidence=0.5, min_tracking_confidence=0.5):
        self.mode = mode
        self.max_num_hand = max_num_hand
        self.min_detection_confidence = min_dectection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode, self.max_num_hand, self.min_detection_confidence, self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmslist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                #print(id, cx, cy)
                lmslist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)
        return lmslist


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    main()
