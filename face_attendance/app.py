from flask import Flask, render_template, request, redirect, url_for, flash, Response, jsonify
from utils.db_utils import init_db, add_user, get_today_attendance
from utils.face_utils import capture_user_images, encode_user_faces, generate_face_recognition_frames
import datetime
import os
import time

app = Flask(__name__)
app.secret_key = "secret_fras_key"

# Shared state for notifications
app.last_event = {"user": None, "timestamp": 0}

# Ensure database is initialized on startup
init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user_id = request.form['user_id'].strip()
        name = request.form['name'].strip()
        
        if not user_id or not name:
            flash("User ID and Name are required!", "error")
            return redirect(url_for('register'))
            
        add_user(user_id, name)
        success = capture_user_images(user_id)
        
        if success:
            encode_user_faces()
            flash(f"User {name} successfully registered!", "success")
        else:
            flash(f"Failed to capture images for {name}.", "error")
            
        return redirect(url_for('index'))
        
    return render_template('register.html')

@app.route('/start_attendance', methods=['GET', 'POST'])
def start_attendance():
    if request.method == 'POST':
        subject = request.form.get('subject', 'General').strip()
        return render_template('attendance.html', subject=subject, streaming=True)
        
    return render_template('attendance.html', streaming=False)

@app.route('/video_feed/<subject>')
def video_feed(subject):
    # Pass the Flask app object to the generator so it can update app.last_event
    return Response(generate_face_recognition_frames(subject, app),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_last_event')
def get_last_event():
    return jsonify(app.last_event)

@app.route('/dashboard')
def dashboard():
    today = datetime.date.today().isoformat()
    attendance_records = get_today_attendance()
    return render_template('dashboard.html', attendance=attendance_records, date=today)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
