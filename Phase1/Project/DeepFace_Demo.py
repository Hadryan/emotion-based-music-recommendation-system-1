import cv2
from deepface import DeepFace

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    result = DeepFace.analyze(frame,actions=['emotion'])
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.1,4)
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    font  = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, result['dominant_emotion'], (200,170), font, 2, (0,0,255), 3,cv2.LINE_8)	
    cv2.imshow("Emotional Detection", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()


# cam = cv2.VideoCapture(0)
# while True:
#     ret_val, img = cam.read()
#     if True: 
#         img = cv2.flip(img, 1)
#     cv2.imshow('my webcam', img)
#     if cv2.waitKey(1) == 27: 
#         break  # esc to quit
# cam.release()
# cv2.destroyAllWindows()