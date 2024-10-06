from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import face_recognition
import pickle
from flask_cors import CORS




with open('face_encodings.pkl', 'rb') as f:
    stored_encodings = pickle.load(f)

app = Flask(__name__)
CORS(app)
img_folder = "photo/"

app.config["img_folder"] = img_folder
app.config["img_retrieve"] = "photos/"

if not os.path.exists(img_folder):
    os.makedirs(img_folder, exist_ok=True)


def process_image(image):
    matches = []
    img = face_recognition.load_image_file(image)
    new_face_encoding = face_recognition.face_encodings(img)
    if new_face_encoding:
        new_face_encoding = new_face_encoding[0]
        for photo_file, encodings in stored_encodings.items():
            results = face_recognition.compare_faces(encodings, new_face_encoding, tolerance=0.6)
            if any(results):
                matches.append(photo_file)

        if not matches:
            return "No matching face found in any stored photo."
    return matches


@app.route('/upload', methods=["POST"])
def home():
    if 'image' not in request.files:
        return jsonify({"message": "Upload a valid file"}), 400

    file = request.files['image']

    try:
        image_path = os.path.join(app.config["img_folder"], file.filename)

        file.save(image_path)
    except Exception as e:
        print(f"Error saving file: {e}")
        return jsonify({"message": "An error occurred while saving the image"}), 500

    matches = process_image(image_path)
    print(matches)
    os.remove(image_path)

    if matches:
        image_urls = [request.host_url + 'images/' + match for match in matches]
        return jsonify({"message": "Matches found", "images": image_urls})
    else:
        return jsonify({"message": "No matches found"})


@app.route('/images/<filename>', methods=['GET', 'POST'])
def get_image(filename):
    try:
        return send_from_directory(app.config["img_retrieve"], filename)
    except Exception as e:
        print(f"Error retrieving image: {e}")
        return jsonify({"message": "Image not found"}), 404


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
