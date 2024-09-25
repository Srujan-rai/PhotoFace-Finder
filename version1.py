import face_recognition
import os
import pickle

# Path to your folder containing images
photo_folder = "photo"
face_encodings = {}

for photo_file in os.listdir(photo_folder):
    if photo_file.endswith(('png', 'jpg', 'jpeg')):
        # Load image
        image_path = os.path.join(photo_folder, photo_file)
        image = face_recognition.load_image_file(image_path)

        # Extract face encodings
        face_locations = face_recognition.face_locations(image)
        encodings = face_recognition.face_encodings(image, face_locations)

        if encodings:
            face_encodings[photo_file] = encodings[0]

with open('face_encodings.pkl', 'wb') as f:
    pickle.dump(face_encodings, f)
