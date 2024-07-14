# from keras.models import load_model
# from keras.models import Model
# from time import sleep
# from keras.preprocessing.image import img_to_array
# from keras.preprocessing import image
# import cv2
# import numpy as np

# face_classifier = cv2.CascadeClassifier(r'C:\Users\dell\Downloads\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\haarcascade_frontalface_default.xml')
# classifier =load_model(r"C:\Users\dell\Downloads\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\model.h5") 


# emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

# cap = cv2.VideoCapture(0)



# while True:
#     _, frame = cap.read()
#     labels = []
#     gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     faces = face_classifier.detectMultiScale(gray)

#     for (x,y,w,h) in faces:
#         cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
#         roi_gray = gray[y:y+h,x:x+w]
#         roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



#         if np.sum([roi_gray])!=0:
#             roi = roi_gray.astype('float')/255.0
#             roi = img_to_array(roi)
#             roi = np.expand_dims(roi,axis=0)

#             prediction = classifier.predict(roi)[0]
#             label=emotion_labels[prediction.argmax()]
#             label_position = (x,y)
#             cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
#         else:
#             cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
#     cv2.imshow('Emotion Detector',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
# from PIL import Image



# # Load face cascade
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Function to classify emotion
# def classify_emotion(image):
#     # Preprocess the image
#     resize_image = cv2.resize(image, (128, 128))
#     input_data = np.expand_dims(resize_image, axis=0) / 255.0
    
#     # Predict emotion
#     pred = classifier.predict(input_data)
#     emotion_label = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
#     result = emotion_label[np.argmax(pred)]
    
#     return result

# # Function for real-time emotion classification with face detection
# def classify_emotion_realtime():
#     cap = cv2.VideoCapture(0)  # You can specify a file path instead of 0 for webcam
    
#     while True:
#         ret, frame = cap.read()
        
#         # Convert frame to grayscale for face detection
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         # Detect faces
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
#         for (x, y, w, h) in faces:
#             # Draw rectangle around detected face
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
#             # Crop face region for emotion classification
#             face_roi = frame[y:y+h, x:x+w]
            
#             # Classify emotion
#             emotion = classify_emotion(face_roi)
            
#             # Display emotion label
#             cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
#         # Display the frame
#         cv2.imshow('Emotion Detection', frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()

# # Run real-time emotion classification
# classify_emotion_realtime()


import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from time import sleep

# Load Haar Cascade Classifier for face detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Load trained emotion detection model
classifier =load_model(r"C:\Users\dell\Downloads\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\model2.h5") 

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Open webcam (change the index if you have multiple cameras)
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        # Draw rectangle around detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Extract the Region of Interest (ROI) and preprocess it
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            # Normalize ROI
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Predict emotion
            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]

            # Display predicted emotion label on the frame
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Emotion Detector', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
