import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0) # Capturing the first camera , hence 0

mpHands = mp.solutions.hands
hands = mpHands.Hands() #Default: Static = "FALSE", max_num_hands = 2, min(Detection,tracking) = 0.5, if goes below 50% then it will detect again.
mpDraw = mp.solutions.drawing_utils

pTime = 0 #previous time set to 0
cTime = 0 #Current time set to 0

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB) # hands only uses rgb color, hence converting to it.
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks) # To check if any of the hands is getting detected.

    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            for id, lm in enumerate(handlms.landmark):
                #print(id,lm) #this prints the position & coordinates x,y,z from the computer vision in decimal points
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h) # converting the decimal x & y points to int pixels.
                print(id, cx, cy)

                #coloring the tips of fingers
                #if id == 0:
                    #cv2.circle(img,(cx,cy), 10, (255, 0, 0), cv2.FILLED)

            mpDraw.draw_landmarks(img ,handlms, mpHands.HAND_CONNECTIONS) # this is to draw the landmarks on our hand, to show the 21 points & the lines connecting them

    cTime = time.time() #Takes the current machine time
    fps = 1/(cTime - pTime) #calculating Fps
    pTime = cTime

    #displaying FPS on Screen
    cv2.putText(img ,str(int(fps)),(10,70), cv2.FONT_HERSHEY_COMPLEX, 3 ,(255,255,255),3)


    cv2.imshow("Hand Detection Program", img)
    cv2.waitKey(1)
