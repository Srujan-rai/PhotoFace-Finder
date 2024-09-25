import face_recognition
import pickle


with open('face_encodings.pkl', 'rb') as f:
    stored_encodings = pickle.load(f)


new_photo_path = "photo/crop.png"
new_image = face_recognition.load_image_file(new_photo_path)


new_face_locations = face_recognition.face_locations(new_image)
new_encodings = face_recognition.face_encodings(new_image, new_face_locations)

if new_encodings:
    new_encoding = new_encodings[0]

  
    for photo_name, encoding in stored_encodings.items():
        results = face_recognition.compare_faces([encoding], new_encoding, tolerance=0.6)

        if results[0]:
            print(f"Match found: {photo_name}")
        else:
            print(f"No match found for {photo_name}")
else:
    print("No faces found in the new image")
