import cv2
import mediapipe as mp
import os
import time
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 1000)
cap.set(10, 150)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpdraw = mp.solutions.drawing_utils

pasttime = 0
brightness = 1.0  

folder = 'colors'
mylist = os.listdir(folder)
overlist = []
col = [0, 0, 255]  
pen_size = 10 
canvas = np.zeros((480, 640, 3), np.uint8)  

for i in mylist:
    image = cv2.imread(f'{folder}/{i}')
    overlist.append(image)

header = overlist[0]
xp, yp = 0, 0
clear_timer = None  

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(img)
    landmarks = []

    if results.multi_hand_landmarks:
        for hn in results.multi_hand_landmarks:
            for id, lm in enumerate(hn.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append([id, cx, cy])
            mpdraw.draw_landmarks(frame, hn, mpHands.HAND_CONNECTIONS)

    if len(landmarks) != 0:
        x1, y1 = landmarks[8][1], landmarks[8][2]
        x2, y2 = landmarks[12][1], landmarks[12][2]

        
        if landmarks[8][2] < landmarks[6][2] and landmarks[12][2] < landmarks[10][2]:
            xp, yp = 0, 0
            if y1 < 100:
                if 71 < x1 < 132:
                    header = overlist[0]
                    col = (0, 0, 255)
                if 132 < x1 < 193:
                    header = overlist[1]
                    col = (0, 0, 0)
                if 193 < x1 < 254:
                    header = overlist[2]
                    col = (0, 255, 0)
                if 254 < x1 < 315:
                    header = overlist[3]
                    col = (157, 0, 255)
                if 315 < x1 < 376:
                    header = overlist[4]
                    col = (0, 255, 255)
                if 376 < x1 < 437:
                    header = overlist[5]
                    pen_size = 10
                if 437 < x1 < 498:
                    header = overlist[6]
                    pen_size = 20

            cv2.rectangle(frame, (x1, y1), (x2, y2), col, cv2.FILLED)

        
        elif landmarks[8][2] < landmarks[6][2]:
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            cv2.line(canvas, (xp, yp), (x1, y1), col, pen_size, cv2.FILLED)
            xp, yp = x1, y1

        
        fingers_extended = sum(landmarks[i][2] < landmarks[i - 2][2] for i in [8, 12, 16, 20]) == 4
        thumb_extended = landmarks[4][1] > landmarks[3][1]  

        if fingers_extended and thumb_extended:
            if clear_timer is None:
                clear_timer = time.time()  
            elif time.time() - clear_timer > 3:  
                canvas = np.zeros((480, 640, 3), np.uint8)
                print("Canvas Cleared")
                clear_timer = None
        else:
            clear_timer = None  

    
    frame = cv2.convertScaleAbs(frame, alpha=brightness, beta=0)

   
    frame[0:95, 0:640] = header

    
    ctime = time.time()
    fps = 1 / (ctime - pasttime)
    pasttime = ctime
    cv2.putText(frame, f'FPS: {int(fps)}', (490, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

    cv2.imshow('cam', frame)
    cv2.imshow('canvas', canvas) 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()