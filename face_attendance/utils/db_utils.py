# Database utilities
import sqlite3
import datetime
import os

# Database file path
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'database.db')

def get_db_connection():
    """Establishes and returns a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Creates the necessary tables if they don't already exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # TABLE 1 — users
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # TABLE 2 — attendance (added 'subject' column)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            subject TEXT,
            date DATE,
            time TIME,
            status TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"[INFO] Database initialized at {DB_PATH}")

def add_user(user_id, name):
    """Adds a new user to the users table."""
    conn = get_db_connection()
    try:
        conn.execute('INSERT INTO users (id, name) VALUES (?, ?)', (user_id, name))
        conn.commit()
        print(f"[SUCCESS] User {name} (ID: {user_id}) added to database.")
    except sqlite3.IntegrityError:
        print(f"[ERROR] User ID {user_id} already exists in database.")
    finally:
        conn.close()

def mark_attendance(user_id, subject="General"):
    """Marks attendance for a user for a specific subject, preventing duplicates for the same day/subject."""
    today = datetime.date.today().isoformat()
    now = datetime.datetime.now().strftime("%H:%M:%S")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if attendance is already marked for this user, date, AND subject
    cursor.execute('SELECT * FROM attendance WHERE user_id=? AND date=? AND subject=?', (user_id, today, subject))
    if cursor.fetchone() is None:
        cursor.execute(
            'INSERT INTO attendance (user_id, subject, date, time, status) VALUES (?, ?, ?, ?, ?)',
            (user_id, subject, today, now, 'Present')
        )
        conn.commit()
        print(f"[SUCCESS] Attendance marked for {user_id} in {subject} at {now}")
        conn.close()
        return True
    else:
        # Already marked for this subject today
        conn.close()
        return False

def get_today_attendance():
    """Returns all attendance records for the current date."""
    today = datetime.date.today().isoformat()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Join with users table to get the user's name
    query = '''
        SELECT a.user_id, u.name, a.subject, a.date, a.time, a.status 
        FROM attendance a
        JOIN users u ON a.user_id = u.id
        WHERE a.date = ?
    '''
    cursor.execute(query, (today,))
    rows = cursor.fetchall()
    conn.close()
    return rows
