from utils.db_utils import init_db, add_user, mark_attendance, get_today_attendance
from utils.attendance_utils import get_today_report
import os
import sys

# Ensure current directory is in path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_db_operations():
    print("--- Testing Database Operations ---")
    
    # 1. Initialize
    init_db()
    
    # 2. Add Users
    print("\n[STEP 1] Adding test users...")
    add_user("test_01", "Alice")
    add_user("test_02", "Bob")
    
    # 3. Mark Attendance
    print("\n[STEP 2] Marking attendance...")
    mark_attendance("test_01")
    mark_attendance("test_01")  # Should be a duplicate, not added again
    mark_attendance("test_02")
    
    # 4. Fetch Attendance
    print("\n[STEP 3] Fetching today's attendance (raw):")
    rows = get_today_attendance()
    for row in rows:
        print(f"ID: {row['user_id']}, Name: {row['name']}, Time: {row['time']}")
        
    # 5. Fetch Report (Pandas)
    print("\n[STEP 4] Fetching today's report (Pandas DataFrame):")
    df = get_today_report()
    print(df)
    
    print("\n--- Database Test Complete ---")

if __name__ == "__main__":
    test_db_operations()
