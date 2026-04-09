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
    
    # TABLE 1 - batches
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_name TEXT NOT NULL,
            subject TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(batch_name, subject)
        )
    ''')
    
    # TABLE 2 - users (students)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id INTEGER NOT NULL,
            enrollment_number TEXT NOT NULL,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (batch_id) REFERENCES batches (id),
            UNIQUE(batch_id, enrollment_number)
        )
    ''')
    
    # TABLE 3 - attendance
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id INTEGER NOT NULL,
            enrollment_number TEXT NOT NULL,
            date DATE NOT NULL,
            time TIME NOT NULL,
            status TEXT NOT NULL,
            FOREIGN KEY (batch_id) REFERENCES batches (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"[INFO] Database initialized at {DB_PATH}")

def create_batch(batch_name, subject):
    """Creates a new batch and returns its ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO batches (batch_name, subject) VALUES (?, ?)', (batch_name, subject))
        conn.commit()
        batch_id = cursor.lastrowid
        print(f"[SUCCESS] Batch '{batch_name}' for subject '{subject}' added (ID: {batch_id}).")
        return batch_id
    except sqlite3.IntegrityError:
        print(f"[ERROR] Batch {batch_name} with subject {subject} already exists.")
        # Fetch the existing batch id
        cursor.execute('SELECT id FROM batches WHERE batch_name=? AND subject=?', (batch_name, subject))
        return cursor.fetchone()['id']
    finally:
        conn.close()

def get_batches():
    """Returns all created batches."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM batches ORDER BY created_at DESC')
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_batch_by_id(batch_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM batches WHERE id=?', (batch_id,))
    row = cursor.fetchone()
    conn.close()
    return row

def add_user(batch_id, enrollment_number, name):
    """Adds a new student to a specific batch."""
    conn = get_db_connection()
    try:
        conn.execute('INSERT INTO users (batch_id, enrollment_number, name) VALUES (?, ?, ?)', (batch_id, enrollment_number, name))
        conn.commit()
        print(f"[SUCCESS] User {name} (Enrollment: {enrollment_number}) added to batch {batch_id}.")
        return True
    except sqlite3.IntegrityError:
        print(f"[ERROR] User {enrollment_number} already exists in batch {batch_id}.")
        return False
    finally:
        conn.close()

def get_users_by_batch(batch_id):
    """Returns all users registered for a specific batch."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE batch_id=? ORDER BY name', (batch_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows

def mark_attendance(batch_id, enrollment_number):
    """Marks attendance for a user in a batch for the current day."""
    today = datetime.date.today().isoformat()
    now = datetime.datetime.now().strftime("%H:%M:%S")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if already marked today
    cursor.execute('SELECT * FROM attendance WHERE batch_id=? AND enrollment_number=? AND date=?', (batch_id, enrollment_number, today))
    if cursor.fetchone() is None:
        cursor.execute(
            'INSERT INTO attendance (batch_id, enrollment_number, date, time, status) VALUES (?, ?, ?, ?, ?)',
            (batch_id, enrollment_number, today, now, 'Present')
        )
        conn.commit()
        print(f"[SUCCESS] Attendance marked for {enrollment_number} in batch {batch_id} at {now}")
        conn.close()
        return True
    else:
        conn.close()
        return False

def get_attendance_summary(batch_id, date=None):
    """Returns all users in a batch with their attendance status for a specific date (defaults to today)."""
    if date is None:
        date = datetime.date.today().isoformat()
        
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Left join to get all users and their attendance if it exists for the given date
    query = '''
        SELECT u.enrollment_number, u.name,
               CASE WHEN a.status IS NOT NULL THEN 'Present' ELSE 'Absent' END as status,
               a.time
        FROM users u
        LEFT JOIN attendance a ON u.enrollment_number = a.enrollment_number 
                               AND a.batch_id = ? 
                               AND a.date = ?
        WHERE u.batch_id = ?
        ORDER BY u.name
    '''
    cursor.execute(query, (batch_id, date, batch_id))
    rows = cursor.fetchall()
    conn.close()
    return rows
