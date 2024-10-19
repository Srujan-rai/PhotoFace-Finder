import sqlite3

def create_database():
    conn = sqlite3.connect('face_encodings.db')
    c = conn.cursor()
    
    # Create table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS encodings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_name TEXT NOT NULL,
                    encoding TEXT NOT NULL
                )''')
    conn.commit()
    conn.close()

# Run this once to create the database and table
create_database()
