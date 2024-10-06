from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import face_recognition
import pandas as pd
import numpy as np
from flask_cors import CORS

# Configuration
img_folder = "photo/"
img_retrieve = "photos/"
excel_file = 'face_encodings.xlsx'

app = Flask(__name__)
CORS(app)

# Flask config
app.config["img_folder"] = img_folder
app.config["img_retrieve"] = img_retrieve

# Ensure the directories exist
if not os.path.exists(img_folder):
    os.makedirs(img_folder, exist_ok=True)

# Load the stored encodings from the Excel file
if os.path.exists(excel_file):
    try:
        df = pd.read_excel(excel_file)
        # Convert the encoding strings back into numpy arrays
        df['encoding'] = df['encoding'].apply(lambda x: np.array(eval(x)))
        print("Loaded encodings from Excel.")
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        df = pd.DataFrame(columns=['filename', 'encoding'])
else:
    df = pd.DataFrame(columns=['filename', 'encoding'])
    print("No Excel file found. Starting fresh.")


def process_image(image):
    """
    Compares the uploaded image's face encoding with stored face encodings.
    Returns a list of matching image filenames.
    """
    matches = []
    img = face_recognition.load_image_file(image)
    new_face_encoding = face_recognition.face_encodings(img)

    if new_face_encoding:
        new_face_encoding = new_face_encoding[0]
        
        # Compare the new encoding with stored encodings
        for _, row in df.iterrows():
            stored_encoding = row['encoding']
            results = face_recognition.compare_faces([stored_encoding], new_face_encoding, tolerance=0.6)
            if any(results):
                matches.append(row['filename'])
        
        if not matches:
            return "No matching face found in any stored photo."
    return matches


@app.route('/upload', methods=["POST"])
def upload_image():
    """
    Handles image uploads, processes the image, and returns matching images.
    """
    if 'image' not in request.files:
        return jsonify({"message": "Upload a valid file"}), 400

    file = request.files['image']

    try:
        # Save the uploaded image temporarily
        image_path = os.path.join(app.config["img_folder"], file.filename)
        file.save(image_path)
    except Exception as e:
        print(f"Error saving file: {e}")
        return jsonify({"message": "An error occurred while saving the image"}), 500

    # Process the uploaded image
    matches = process_image(image_path)
    os.remove(image_path)  # Remove the temporary uploaded image

    if matches:
        image_urls = [request.host_url + 'images/' + match for match in matches]
        return jsonify({"message": "Matches found", "images": image_urls})
    else:
        return jsonify({"message": "No matches found"})


@app.route('/images/<filename>', methods=['GET', 'POST'])
def get_image(filename):
    """
    Serves the requested image from the folder.
    """
    try:
        return send_from_directory(app.config["img_retrieve"], filename)
    except Exception as e:
        print(f"Error retrieving image: {e}")
        return jsonify({"message": "Image not found"}), 404


@app.route('/')
def index():
    """
    Serves the index page.
    """
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
