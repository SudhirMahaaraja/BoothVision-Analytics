import sqlite3

DB_FILE = "backend/database/booth_analytics.db"

# Ensure the path exists before creating the database
import os
os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)

# Initialize the database
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

# Create the visitors table
cursor.execute("""
CREATE TABLE IF NOT EXISTS visitors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    unique_id TEXT,
    gender TEXT,
    age INTEGER,
    timestamp TEXT
);
""")
conn.commit()
conn.close()
print("Database initialized successfully.")
