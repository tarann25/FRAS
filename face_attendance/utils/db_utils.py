import psycopg2
import psycopg2.extras
import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database connection string from Supabase
DATABASE_URL = os.environ.get("DATABASE_URL")

def get_db_connection():
    """Establishes and returns a connection to the PostgreSQL database."""
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is not set. Please check your .env file.")
    
    conn = psycopg2.connect(DATABASE_URL)
    return conn

def init_db():
    """Creates the necessary tables in PostgreSQL if they don't already exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # TABLE 1 - batches
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS batches (
            id SERIAL PRIMARY KEY,
            teacher_id UUID NOT NULL,
            batch_name TEXT NOT NULL,
            subject TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(teacher_id, batch_name, subject)
        )
    ''')
    
    # TABLE 2 - users (students)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            batch_id INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
            enrollment_number TEXT NOT NULL,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(batch_id, enrollment_number)
        )
    ''')
    
    # TABLE 3 - attendance
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id SERIAL PRIMARY KEY,
            batch_id INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
            enrollment_number TEXT NOT NULL,
            date DATE NOT NULL,
            time TIME NOT NULL,
            status TEXT NOT NULL
        )
    ''')
    
    conn.commit()
    cursor.close()
    conn.close()
    print("[INFO] PostgreSQL Database initialized successfully.")

def create_batch(batch_name, subject, teacher_id):
    """Creates a new batch for a specific teacher and returns its ID."""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        cursor.execute(
            'INSERT INTO batches (teacher_id, batch_name, subject) VALUES (%s, %s, %s) RETURNING id', 
            (teacher_id, batch_name, subject)
        )
        conn.commit()
        batch_id = cursor.fetchone()['id']
        print(f"[SUCCESS] Batch '{batch_name}' for subject '{subject}' added by {teacher_id}.")
        return batch_id
    except psycopg2.IntegrityError:
        conn.rollback()
        print(f"[ERROR] Batch {batch_name} with subject {subject} already exists for this teacher.")
        # Fetch the existing batch id
        cursor.execute('SELECT id FROM batches WHERE teacher_id=%s AND batch_name=%s AND subject=%s', (teacher_id, batch_name, subject))
        return cursor.fetchone()['id']
    finally:
        cursor.close()
        conn.close()

def get_batches(teacher_id):
    """Returns all created batches for a specific teacher."""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cursor.execute('SELECT * FROM batches WHERE teacher_id=%s ORDER BY created_at DESC', (teacher_id,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

def get_batch_by_id(batch_id):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cursor.execute('SELECT * FROM batches WHERE id=%s', (batch_id,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return row

def add_user(batch_id, enrollment_number, name):
    """Adds a new student to a specific batch."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            'INSERT INTO users (batch_id, enrollment_number, name) VALUES (%s, %s, %s)', 
            (batch_id, enrollment_number, name)
        )
        conn.commit()
        print(f"[SUCCESS] User {name} (Enrollment: {enrollment_number}) added to batch {batch_id}.")
        return True
    except psycopg2.IntegrityError:
        conn.rollback()
        print(f"[ERROR] User {enrollment_number} already exists in batch {batch_id}.")
        return False
    finally:
        cursor.close()
        conn.close()

def get_users_by_batch(batch_id):
    """Returns all users registered for a specific batch."""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cursor.execute('SELECT * FROM users WHERE batch_id=%s ORDER BY name', (batch_id,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

def mark_attendance(batch_id, enrollment_number):
    """Marks attendance for a user in a batch for the current day."""
    today = datetime.date.today()
    now = datetime.datetime.now().time()
    
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    # Check if already marked today
    cursor.execute(
        'SELECT * FROM attendance WHERE batch_id=%s AND enrollment_number=%s AND date=%s', 
        (batch_id, enrollment_number, today)
    )
    if cursor.fetchone() is None:
        cursor.execute(
            'INSERT INTO attendance (batch_id, enrollment_number, date, time, status) VALUES (%s, %s, %s, %s, %s)',
            (batch_id, enrollment_number, today, now, 'Present')
        )
        conn.commit()
        print(f"[SUCCESS] Attendance marked for {enrollment_number} in batch {batch_id} at {now}")
        cursor.close()
        conn.close()
        return True
    else:
        cursor.close()
        conn.close()
        return False

def get_attendance_summary(batch_id, date=None):
    """Returns all users in a batch with their attendance status for a specific date (defaults to today)."""
    if date is None:
        date = datetime.date.today()
        
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    # Left join to get all users and their attendance if it exists for the given date
    query = '''
        SELECT u.enrollment_number, u.name,
               CASE WHEN a.status IS NOT NULL THEN 'Present' ELSE 'Absent' END as status,
               a.time
        FROM users u
        LEFT JOIN attendance a ON u.enrollment_number = a.enrollment_number 
                               AND a.batch_id = %s 
                               AND a.date = %s
        WHERE u.batch_id = %s
        ORDER BY u.name
    '''
    cursor.execute(query, (batch_id, date, batch_id))
    rows = cursor.fetchall()
    
    # Format times for display if present
    formatted_rows = []
    for row in rows:
        row_dict = dict(row)
        if row_dict['time']:
            # Format time to HH:MM:SS string for frontend
            row_dict['time'] = row_dict['time'].strftime("%H:%M:%S")
        formatted_rows.append(row_dict)
        
    cursor.close()
    conn.close()
    return formatted_rows
