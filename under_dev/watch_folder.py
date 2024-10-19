import os
import face_recognition
import sqlite3
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time

# Configuration for folders
img_folder = "watched_folder/"
retrieval_folder = "retrieval/"  # Add this folder for retrieval of encoded images

# Create the database and the 'encodings' table if it doesn't exist
def initialize_database():
    conn = sqlite3.connect('face_encodings.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS encodings (
                    image_name TEXT PRIMARY KEY, 
                    encoding TEXT)''')
    conn.commit()
    conn.close()

# Function to encode the new image and save it in the database
def encode_image(image_path):
    conn = sqlite3.connect('face_encodings.db')
    c = conn.cursor()

    filename = os.path.basename(image_path)
    
    # Check if the image is already in the database
    c.execute("SELECT image_name FROM encodings WHERE image_name=?", (filename,))
    if c.fetchone():
        print(f"Skipping {filename}, already encoded.")
        conn.close()
        return

    # Load and process the image
    img = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(img)
    
    if face_encodings:
        new_encoding = face_encodings[0]
        # Convert the encoding to a comma-separated string for storage
        encoding_str = ','.join(map(str, new_encoding))

        # Insert the new encoding into the database
        c.execute("INSERT INTO encodings (image_name, encoding) VALUES (?, ?)", (filename, encoding_str))
        conn.commit()
        print(f"Encoded and saved {filename}")

        # Move the encoded image to the retrieval folder
        retrieval_path = os.path.join(retrieval_folder, filename)
        os.rename(image_path, retrieval_path)
        print(f"Moved {filename} to {retrieval_folder}")
    else:
        print(f"No face found in {filename}")

    conn.close()

# Function to process all existing images in the watched folder
def process_existing_images():
    for filename in os.listdir(img_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(img_folder, filename)
            print(f"Processing existing image: {image_path}")
            encode_image(image_path)

# Event handler class to respond to new files being added to the folder
class NewImageHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            # Check if the file is an image (you can extend this to check specific formats if needed)
            if event.src_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"New image detected: {event.src_path}")
                encode_image(event.src_path)

# Watch the folder for new images
def watch_folder():
    event_handler = NewImageHandler()
    observer = Observer()
    observer.schedule(event_handler, img_folder, recursive=False)
    observer.start()
    print(f"Watching folder: {img_folder}")
    
    try:
        while True:
            time.sleep(1)  # Keep the script running
    except KeyboardInterrupt:
        observer.stop()

    observer.join()

if __name__ == "__main__":
    # Create the folders if they do not exist
    if not os.path.exists(img_folder):
        os.makedirs(img_folder, exist_ok=True)
    if not os.path.exists(retrieval_folder):
        os.makedirs(retrieval_folder, exist_ok=True)

    # Initialize the database
    initialize_database()

    # Process existing images in the watched folder
    process_existing_images()

    # Start watching the folder for new images
    watch_folder()
