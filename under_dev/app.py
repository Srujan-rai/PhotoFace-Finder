from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import face_recognition
import sqlite3
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
img_folder = "photo/"
db_file = "face_encodings.db"  # SQLite database file

app.config["img_folder"] = img_folder
app.config["img_retrieve"] = "photos/"

# Ensure the upload folder exists
if not os.path.exists(img_folder):
    os.makedirs(img_folder, exist_ok=True)

# Initialize the database and create the necessary table if it doesn't exist
def init_db():
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    # Create table if it doesn't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS encodings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_name TEXT NOT NULL,
        encoding TEXT NOT NULL
    )
    ''')
    conn.commit()
    conn.close()

# Load stored encodings from the database
def load_encodings():
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("SELECT image_name, encoding FROM encodings")
    rows = c.fetchall()
    conn.close()

    stored_encodings = {}
    for row in rows:
        image_name = row[0]
        # Convert the stored string back to a list of floats
        encoding = list(map(float, row[1].split(',')))
        stored_encodings[image_name] = encoding
    return stored_encodings

# Process uploaded image and find matches
def process_image(image):
    matches = []
    img = face_recognition.load_image_file(image)
    new_face_encoding = face_recognition.face_encodings(img)

    if new_face_encoding:
        new_face_encoding = new_face_encoding[0]
        stored_encodings = load_encodings()  # Load encodings from the database

        # Compare uploaded image encoding with stored encodings
        for image_name, stored_encoding in stored_encodings.items():
            results = face_recognition.compare_faces([stored_encoding], new_face_encoding, tolerance=0.6)
            if any(results):
                matches.append(image_name)
                

        if not matches:
            return "No matching face found in any stored photo."

    return matches

# Upload image and check for face matches
@app.route('/upload', methods=["POST"])
def home():
    if 'image' not in request.files:
        return jsonify({"message": "Upload a valid file"}), 400

    file = request.files['image']
    image_path = os.path.join(app.config["img_folder"], file.filename)

    try:
        # Save the uploaded image
        file.save(image_path)
    except Exception as e:
        print(f"Error saving file: {e}")
        return jsonify({"message": "An error occurred while saving the image"}), 500

    # Process the image and find matches
    matches = process_image(image_path)
    os.remove(image_path)  # Clean up after processing

    if matches:
        image_urls = [request.host_url + 'images/' + match for match in matches]
        print(image_urls)
        return jsonify({"message": "Matches found", "images": image_urls})
    else:
        return jsonify({"message": "No matches found"})

# Serve images
@app.route('/images/<filename>', methods=['GET'])
def get_image(filename):
    try:
        return send_from_directory(app.config["img_retrieve"], filename)
    except Exception as e:
        print(f"Error retrieving image: {e}")
        return jsonify({"message": "Image not found"}), 404

# Home route to serve the HTML template
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    init_db()  # Initialize the database on startup
    app.run(debug=True, host="0.0.0.0")
