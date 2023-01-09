import cv2
import mediapipe as mp
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.models import Sequential, load_model

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

#Define Path
model_path = './models/model.h5'
model_weights_path = './models/weights.h5'
test_path = 'dataset/testing'

#Load the pre-trained models
model = load_model(model_path)
model.load_weights(model_weights_path)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_hand = hands.process(image)
    br,kl,_ = image.shape
    black_image = np.zeros((br, kl, 3), dtype = "uint8")
    lRes_hand = np.zeros([21,3])
    
    _, frame = cap.read()
    frame = np.zeros((150, 150, 3), dtype = "uint8")
    #frame = img_to_array(frame)
    #frame = np.expand_dims(frame, axis=0)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results_hand.multi_hand_landmarks:
        for hand_landmarks in results_hand.multi_hand_landmarks:
            for i in range(21):
                lRes_hand[i,0]=hand_landmarks.landmark[i].x*kl;
                lRes_hand[i,1]=hand_landmarks.landmark[i].y*br;
                
            minx_hand = min(lRes_hand[:,0]) - 20
            miny_hand = min(lRes_hand[:,1]) - 20
            
            maxx_hand = max(lRes_hand[:,0]) + 20
            maxy_hand = max(lRes_hand[:,1]) + 20
            
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            cv2.rectangle (image, (int (minx_hand), int (miny_hand)), (int (maxx_hand), int (maxy_hand)), (0,0,0), 2)
            cv2.rectangle (frame, (int (minx_hand), int (miny_hand)), (int (maxx_hand), int (maxy_hand)), (255,255,255), 2)   
           
            array = model.predict(frame)
            result = array[0]
            #print(result)
            answer = np.argmax(result)
            if answer == 0:
              print("Predicted: berhenti")
            elif answer == 1:
              print("Predicted: mengepal")
           
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    # cv2.imshow('Black Image', cv2.flip(black_image, 1))
    cv2.imshow('Prediction', cv2.flip(frame, 1))
    if cv2.waitKey(5) & 0xFF == 27:
        #cap.release()
        #cv2.destroyAllWindows()
        break
cap.release()
cv2.destroyAllWindows()