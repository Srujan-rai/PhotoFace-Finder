import face_recognition
import pickle

# Load the saved face encodings from stored photos
with open('face_encodings.pkl', 'rb') as f:
    stored_encodings = pickle.load(f)

new_photo_path = "photos/WhatsApp Image 2024-09-25 at 10.17.52 PM.jpeg"
new_image = face_recognition.load_image_file(new_photo_path)

new_face_encodings = face_recognition.face_encodings(new_image)

if new_face_encodings:
    new_face_encoding = new_face_encodings[0]  

    match_found = False

    for photo_name, encodings in stored_encodings.items():
        results = face_recognition.compare_faces(encodings, new_face_encoding, tolerance=0.6)

        if any(results):
            print(f"Match found in stored photo: {photo_name}")
            match_found = True

    if not match_found:
        print("No matching face found in any stored photo.")
else:
    print("No face found in the uploaded image.")
