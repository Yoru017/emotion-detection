from transformers import pipeline
import cv2 as cv
import numpy as np
from PIL import Image


face = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')


expression_model = pipeline("image-classification", model="trpakov/vit-face-expression")

# Warna berdasarkan ekspresi
expression_colors = {
    "happy": (0, 255, 0),       # Hijau
    "sad": (255, 0, 0),         # Biru
    "angry": (0, 0, 255),       # Merah
    "surprise": (255, 255, 0), # Kuning
    "disgust": (255, 100, 0),    # Biru
    "fear": (255, 0, 0),         # Biru
    "neutral": (255, 255, 255)  # Putih
}


cam = cv.VideoCapture(0)


cam.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    
    ret, frame = cam.read()

    if not ret:
        break

    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    
    faces = face.detectMultiScale(gray, scaleFactor=1.1, 
                                          minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        
        face_img = frame[y:y+h, x:x+w]

        
        face_img_rgb = cv.cvtColor(face_img, cv.COLOR_BGR2RGB)

        
        pil_image = Image.fromarray(face_img_rgb)

        
        expression_preds = expression_model(pil_image)
        predicted_expression = expression_preds[0]['label']

        
        cv.rectangle(frame, (x, y), (x+w, y+h), expression_colors.get(predicted_expression, (0, 0, 255)), 2)
        cv.putText(frame, f"{predicted_expression}", (x, y-10), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.9, 
                    expression_colors.get(predicted_expression, (0, 0, 255)), 2)

    
    cv.imshow('Deteksi Ekspresi Wajah', frame)

    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


cam.release()
cv.destroyAllWindows()
