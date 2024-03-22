import cv2
import numpy as np
import mediapipe as mp
import time

cap = cv2.VideoCapture("video2.mp4")
mpHands = mp.solutions.hands

hands=mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
pTime=0
cTime=0
while True:
    success, img = cap.read()
    new_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    results=hands.process(new_img)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)

            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)

                h , w, c = img.shape

                cx , cy = int(lm.x*w), int(lm.y*h)

                #serce parmak ucu -- işaret ediyoruz.
                if id ==8:
                    cv2.circle(img,(cx,cy),9,(255,0,0),cv2.FILLED)



    #fps
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,"FPS:"+str(int(fps)),(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break