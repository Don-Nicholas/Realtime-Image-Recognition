import numpy as np
from keras.models import model_from_json
import operator
import cv2
import sys, os

# Loading the model
json_file = open("musa-acuminata.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("musa-acuminata-bw.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(1)
classes = ['freshripe', 'freshunripe', 'overripe', 'ripe', 'rotten', 'unripe']

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    # Got this from collect-data.py
    # Coordinates of the ROI
    # x1 = int(0.5*frame.shape[1])
    x1 = 200
    y1 = 20
    x2 = frame.shape[1]-20
    y2 = int(0.70*frame.shape[1])
    # y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    
    # Resizing the ROI so it can be fed to the model for prediction
    roi = cv2.resize(roi, (150, 150))
    result = loaded_model.predict(roi.reshape(1, 150, 150, 3))
    val = int(np.argmax(result))
    res = classes[val]
    
    cv2.putText(frame, res, (int(((x1+x2)/2)-20), (y2 + 20)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255), 1)    
    cv2.putText(frame, "Classes:", ((20), (50)), cv2.FONT_HERSHEY_COMPLEX, 0.85, (0,255,255), 1) 
    
    cv2.putText(frame, classes[0], ((20), (100)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255), 1)
    cv2.putText(frame, classes[1], ((20), (130)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255), 1)
    cv2.putText(frame, classes[2], ((20), (160)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255), 1)
    cv2.putText(frame, classes[3], ((20), (190)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255), 1)
    cv2.putText(frame, classes[4], ((20), (220)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255), 1)
    cv2.putText(frame, classes[5], ((20), (250)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255), 1)
    
    cv2.imshow("Realtime Image Recognition", frame)
    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
        
 
cap.release()
cv2.destroyAllWindows()