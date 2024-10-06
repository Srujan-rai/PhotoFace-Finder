import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
import face_recognition
import pandas as pd
import numpy as np
import logging

# Configuration
PHOTO_FOLDER = "photos"
EXCEL_FILE = "under_dev/face_encodings.xlsx"
LOG_FILE = "under_dev/photo_face_finder.log"

# Setup logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Ensure directories exist
os.makedirs(PHOTO_FOLDER, exist_ok=True)
os.makedirs("temp", exist_ok=True)

# Initialize or load the Excel file
if os.path.exists(EXCEL_FILE):
    try:
        df = pd.read_excel(EXCEL_FILE)
        # Convert encoding lists back to numpy arrays
        df['encoding'] = df['encoding'].apply(lambda x: np.array(eval(x)))
        logging.info("Loaded existing encodings from Excel.")
    except Exception as e:
        logging.error(f"Failed to load Excel file: {e}")
        df = pd.DataFrame(columns=['filename', 'encoding'])
else:
    df = pd.DataFrame(columns=['filename', 'encoding'])
    logging.info("Initialized new DataFrame for encodings.")

def process_existing_photos(dataframe, folder, excel_path):
    """
    Process all existing photos in the folder that are not yet in the DataFrame.
    """
    processed_files = set(dataframe['filename'].tolist())
    new_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f not in processed_files]
    
    if not new_files:
        logging.info("No new existing photos to process.")
        return dataframe
    
    logging.info(f"Processing {len(new_files)} existing photos.")
    
    for filename in new_files:
        filepath = os.path.join(folder, filename)
        try:
            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)
            
            if encodings:
                encoding = encodings[0]  # Assuming one face per photo
                dataframe = dataframe.append({
                    'filename': filename,
                    'encoding': encoding.tolist()  # Convert numpy array to list for Excel
                }, ignore_index=True)
                logging.info(f"Encoded and added existing photo: {filename}")
            else:
                logging.warning(f"No face found in existing photo: {filename}.")
        except Exception as e:
            logging.error(f"Error processing existing photo {filename}: {e}")
    
    # Save updated DataFrame to Excel
    dataframe.to_excel(excel_path, index=False)
    logging.info("Finished processing existing photos.")
    return dataframe

class NewPhotoHandler(FileSystemEventHandler):
    def __init__(self, dataframe, excel_path):
        super().__init__()
        self.df = dataframe
        self.excel_path = excel_path
        self.processed_files = set(self.df['filename'].tolist())

    def on_created(self, event):
        if not event.is_directory:
            filepath = event.src_path
            filename = os.path.basename(filepath)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                if filename in self.processed_files:
                    logging.info(f"Duplicate photo detected: {filename}. Skipping.")
                    return
                logging.info(f"New photo detected: {filename}")
                try:
                    # Encode the face
                    image = face_recognition.load_image_file(filepath)
                    encodings = face_recognition.face_encodings(image)
                    
                    if encodings:
                        encoding = encodings[0]  # Assuming one face per photo
                        # Append to the DataFrame
                        self.df = self.df.append({
                            'filename': filename,
                            'encoding': encoding.tolist()  # Convert numpy array to list for Excel
                        }, ignore_index=True)
                        # Save to Excel
                        self.df.to_excel(self.excel_path, index=False)
                        self.processed_files.add(filename)
                        logging.info(f"Encoded and saved: {filename}")
                    else:
                        logging.warning(f"No face found in {filename}.")
                except Exception as e:
                    logging.error(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # Process existing photos first
    df = process_existing_photos(df, PHOTO_FOLDER, EXCEL_FILE)
    
    # Initialize event handler with updated DataFrame
    event_handler = NewPhotoHandler(df, EXCEL_FILE)
    observer = Observer()
    observer.schedule(event_handler, PHOTO_FOLDER, recursive=False)
    observer.start()
    logging.info(f"Started monitoring {PHOTO_FOLDER} for new photos.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logging.info("Stopped folder monitoring.")
    observer.join()
