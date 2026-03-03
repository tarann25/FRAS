# Attendance logic utilities
import pandas as pd
from .db_utils import mark_attendance as mark_att_db, get_today_attendance

def mark_attendance(user_id, subject="General"):
    """Higher-level mark attendance function."""
    return mark_att_db(user_id, subject)

def get_today_report():
    """Returns today's attendance as a formatted DataFrame."""
    rows = get_today_attendance()
    if not rows:
        return pd.DataFrame(columns=['User ID', 'Name', 'Subject', 'Date', 'Time', 'Status'])
    
    df = pd.DataFrame(rows, columns=['User ID', 'Name', 'Subject', 'Date', 'Time', 'Status'])
    return df
