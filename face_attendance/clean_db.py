import os
import shutil
from utils.db_utils import get_db_connection, init_db
from utils.face_utils import supabase, BUCKET_NAME

def clean_database():
    print("[INFO] Cleaning PostgreSQL Database...")
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # CASCADE ensures related records in users and attendance are also deleted
        # RESTART IDENTITY resets the auto-incrementing primary keys back to 1
        cursor.execute("TRUNCATE batches, users, attendance RESTART IDENTITY CASCADE;")
        conn.commit()
        print("[SUCCESS] All tables truncated successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to clean database: {e}")
    finally:
        cursor.close()
        conn.close()

def clean_local_files():
    print("[INFO] Cleaning local dataset and encodings folders...")
    for folder in ['dataset', 'encodings']:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"[SUCCESS] Deleted local '{folder}' folder.")
        else:
            print(f"[INFO] '{folder}' folder does not exist locally.")

def clean_supabase_bucket():
    if not supabase:
        print("[WARNING] Supabase client not initialized. Skipping bucket cleanup.")
        return
        
    print(f"[INFO] Emptying Supabase bucket '{BUCKET_NAME}'...")
    try:
        # List all files in the bucket
        res = supabase.storage.from_(BUCKET_NAME).list()
        if res:
            files_to_delete = [file['name'] for file in res if file['name'] != '.emptyFolderPlaceholder']
            if files_to_delete:
                supabase.storage.from_(BUCKET_NAME).remove(files_to_delete)
                print(f"[SUCCESS] Deleted {len(files_to_delete)} files from Supabase bucket.")
            else:
                print("[INFO] Supabase bucket is already empty.")
    except Exception as e:
        print(f"[ERROR] Failed to clean Supabase bucket: {e}")

if __name__ == '__main__':
    clean_database()
    clean_local_files()
    clean_supabase_bucket()
    print("\n[SUCCESS] Entire system has been reset to a fresh state!")