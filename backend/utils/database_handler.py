import sqlite3
from datetime import datetime


# Database connection setup
def connect_db():
    conn = sqlite3.connect('backend/database/database.db')
    return conn


# Create tables
def create_tables():
    conn = connect_db()
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS visitors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    unique_id TEXT NOT NULL,
                    gender TEXT,
                    age INTEGER,
                    timestamp TEXT,
                    emotion TEXT)''')

    conn.commit()
    conn.close()


# Save visitor details to the database
def save_visitor_details(unique_id, gender, age, emotion):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = connect_db()
    c = conn.cursor()

    c.execute('''INSERT INTO visitors (unique_id, gender, age, timestamp, emotion) 
                 VALUES (?, ?, ?, ?, ?)''',
              (unique_id, gender, age, timestamp, emotion))

    conn.commit()
    conn.close()


# Example of fetching all visitor details (could be used in Flask later)
def get_all_visitors():
    conn = connect_db()
    c = conn.cursor()
    c.execute('SELECT * FROM visitors')
    visitors = c.fetchall()
    conn.close()
    return visitors
