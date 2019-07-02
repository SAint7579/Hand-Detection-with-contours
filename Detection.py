import numpy as np
import cv2

def blackout(frame,gray):
    '''
    Removes the face from the passed frame using haar cascades
    '''
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray,1.2,1)
    for (x,y,w,h) in faces:
        #Blacking out the face
        frame[y:y+h+50,x:x+w] = 0
    return frame

#To capture from primary webcam
cap = cv2.VideoCapture(0)

while True:
    _,frame = cap.read()
    frame_wof=np.copy(frame)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #Removing the face
    frame_wof = blackout(frame_wof,gray)
    #Converting the color scheme
    hsv = cv2.cvtColor(cv2.medianBlur(frame_wof,15),cv2.COLOR_BGR2HSV)
    
    lower = np.array([0, 10, 60])     #Lower range of HSV color
    upper = np.array([40, 165, 255])  #Upper range of HSV color
    #Creating the mask
    mask = cv2.inRange(hsv,lower,upper)
    #Removing noise form the mask
    mask = cv2.dilate(mask,None,iterations=2)
    
    #Extracting contours from the mask
    cnts,_ = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        for c in cnts:
            #To prevent the detection of the noise
            if cv2.contourArea(c) > 8000:
                #Fixing covex defect
                hull = cv2.convexHull(c)
                #Drawing the contours
                cv2.drawContours(frame,[hull],0,(0,0,255),2)
                #Creating the bounding rectangel
                x,y,w,h = cv2.boundingRect(hull)
                cv2.rectangle(frame,(x-10,y-10),(x+w+10,y+h+10),(0,255,0),2)
    #Showing the mask
    cv2.imshow("Video",frame)
    cv2.imshow("Mask",mask)

    if cv2.waitKey(1)& 0xFF == ord('q'):
        break
#Releasing the camera
cap.release()
cv2.destroyAllWindows()
    
