from flask import Flask, render_template, request, redirect, url_for, flash, Response, jsonify
from utils.db_utils import init_db, create_batch, get_batches, get_batch_by_id, add_user, get_users_by_batch, get_attendance_summary
from utils.face_utils import encode_batch_faces, generate_face_recognition_frames, generate_registration_frames, CAPTURE_REQUESTS
import os
import time
import uuid

app = Flask(__name__)
app.secret_key = "secret_fras_key"

# Shared state for notifications
app.last_event = {"enrollment": None, "timestamp": 0}

# Ensure database is initialized on startup
init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register_batch', methods=['GET', 'POST'])
def register_batch():
    if request.method == 'POST':
        batch_name = request.form['batch_name'].strip()
        subject = request.form['subject'].strip()
        
        if not batch_name or not subject:
            flash("Batch Name and Subject are required!", "error")
            return redirect(url_for('register_batch'))
            
        batch_id = create_batch(batch_name, subject)
        return redirect(url_for('register_students', batch_id=batch_id))
        
    return render_template('register_batch.html')

@app.route('/register_students/<int:batch_id>')
def register_students(batch_id):
    batch = get_batch_by_id(batch_id)
    if not batch:
        flash("Batch not found", "error")
        return redirect(url_for('index'))
    return render_template('register_students.html', batch=batch)

@app.route('/registration_video_feed/<batch_name>/<subject>')
def registration_video_feed(batch_name, subject):
    return Response(generate_registration_frames(batch_name, subject),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_student', methods=['POST'])
def capture_student():
    data = request.json
    batch_id = data.get('batch_id')
    batch_name = data.get('batch_name')
    subject = data.get('subject')
    enrollment = data.get('enrollment')
    name = data.get('name')

    if not all([batch_id, batch_name, subject, enrollment, name]):
        return jsonify({"success": False, "error": "Missing data"})

    # Add user to database
    if not add_user(batch_id, enrollment, name):
        return jsonify({"success": False, "error": "User already exists or DB error"})

    # Create capture request
    req_key = str(uuid.uuid4())
    CAPTURE_REQUESTS[req_key] = {
        'batch_name': batch_name,
        'subject': subject,
        'enrollment': enrollment,
        'count': 0,
        'done': False
    }

    return jsonify({"success": True, "req_key": req_key})

@app.route('/capture_status/<req_key>')
def capture_status(req_key):
    req = CAPTURE_REQUESTS.get(req_key)
    if not req:
        return jsonify({"error": "Not found"}), 404
    
    return jsonify({
        "count": req['count'],
        "done": req['done']
    })

@app.route('/complete_registration/<int:batch_id>', methods=['POST'])
def complete_registration(batch_id):
    batch = get_batch_by_id(batch_id)
    if encode_batch_faces(batch_id, batch['batch_name'], batch['subject']):
        flash(f"Batch {batch['batch_name']} registered successfully!", "success")
    else:
        flash(f"Failed to encode faces. Were any students added?", "error")
    return redirect(url_for('registration_summary', batch_id=batch_id))

@app.route('/registration_summary/<int:batch_id>')
def registration_summary(batch_id):
    batch = get_batch_by_id(batch_id)
    users = get_users_by_batch(batch_id)
    return render_template('registration_summary.html', batch=batch, count=len(users))

@app.route('/mark_attendance')
def mark_attendance():
    batches = get_batches()
    return render_template('select_batch.html', batches=batches)

@app.route('/attendance_viewfinder/<int:batch_id>')
def attendance_viewfinder(batch_id):
    batch = get_batch_by_id(batch_id)
    return render_template('attendance_viewfinder.html', batch=batch)

@app.route('/attendance_video_feed/<int:batch_id>')
def attendance_video_feed(batch_id):
    return Response(generate_face_recognition_frames(batch_id, app),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_last_event')
def get_last_event():
    return jsonify(app.last_event)

@app.route('/attendance_summary/<int:batch_id>')
def attendance_summary(batch_id):
    batch = get_batch_by_id(batch_id)
    summary = get_attendance_summary(batch_id)
    return render_template('attendance_summary.html', batch=batch, summary=summary)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
