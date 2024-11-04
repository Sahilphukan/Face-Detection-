import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.applications.inception_v3 import preprocess_input

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
def load_model():
    base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    model = Model(inputs=base_model.input, outputs=base_model.output)
    return model
model = load_model()
def preprocess_face(face_image):
    face_image = cv2.resize(face_image, (160, 160))  
    face_image = kimage.img_to_array(face_image)
    face_image = np.expand_dims(face_image, axis=0)
    face_image = preprocess_input(face_image)
    return face_image
def extract_face_features(model, face_image):
    preprocessed_face = preprocess_face(face_image)
    features = model.predict(preprocessed_face)
    return features
cap = cv2.VideoCapture(0)  
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()
with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                face_image = rgb_frame[y:y+h, x:x+w]
                if face_image.size == 0:
                    continue
                face_features = extract_face_features(model, face_image)
                print("Face features shape:", face_features.shape)                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)             
                label = 'Face Detected'
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                text_w, text_h = text_size         
                box_coords = ((x, y - text_h - 10), (x + text_w, y))
                cv2.rectangle(frame, box_coords[0], box_coords[1], (0, 0, 0), -1)         
                cv2.putText(frame, label, (x, y - 5), font, font_scale, (255, 255, 255), font_thickness)
        cv2.imshow('Face Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
