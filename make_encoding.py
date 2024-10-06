import face_recognition
import os
import pickle

photo_folder = "photos"
face_encodings = {}

for photo_file in os.listdir(photo_folder):
    
    if photo_file.endswith(('png', 'jpg', 'jpeg')):
        
        image_path = os.path.join(photo_folder, photo_file)
        image = face_recognition.load_image_file(image_path)

        encodings = face_recognition.face_encodings(image)

        if encodings:
            face_encodings[photo_file] = encodings

with open('face_encodings.pkl', 'wb') as f:
    pickle.dump(face_encodings, f)
