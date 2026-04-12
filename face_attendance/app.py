import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from flask import Flask, render_template, request, redirect, url_for, flash, Response, jsonify, session
from functools import wraps
from utils.db_utils import init_db, create_batch, get_batches, get_batch_by_id, add_user, get_users_by_batch, get_attendance_summary
from utils.face_utils import encode_batch_faces, generate_face_recognition_frames, generate_registration_frames, CAPTURE_REQUESTS, supabase
import os
import time
import uuid

app = Flask(__name__)
app.secret_key = "secret_fras_key_super_secure"

# Shared state for notifications
app.last_event = {"enrollment": None, "timestamp": 0}

# Ensure database is initialized on startup
init_db()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # In a full production app you would verify the JWT here. 
        # For this setup, we rely on the Supabase SDK session stored in Flask's secure cookie.
        if 'user' not in session:
            flash("Please log in to access this page.", "error")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        try:
            res = supabase.auth.sign_in_with_password({"email": email, "password": password})
            
            # Extract name from metadata if it exists, otherwise fallback to email
            name = email
            if hasattr(res.user, 'user_metadata') and res.user.user_metadata:
                name = res.user.user_metadata.get('full_name', email)
                
            session['user'] = {'id': res.user.id, 'email': res.user.email, 'name': name}
            flash(f"Welcome back, {name}!", "success")
            return redirect(url_for('index'))
        except Exception as e:
            flash(f"Invalid email or password. {e}", "error")
            
    return render_template('login.html')

@app.route('/register_teacher', methods=['GET', 'POST'])
def register_teacher():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        try:
            res = supabase.auth.sign_up({
                "email": email, 
                "password": password,
                "options": {
                    "data": {
                        "full_name": name
                    }
                }
            })
            flash("Registration successful! You can now log in.", "success")
            return redirect(url_for('login'))
        except Exception as e:
            flash(f"Registration failed: {e}", "error")
            
    return render_template('register_teacher.html')

@app.route('/logout')
def logout():
    try:
        supabase.auth.sign_out()
    except Exception:
        pass
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for('login'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register_batch', methods=['GET', 'POST'])
@login_required
def register_batch():
    if request.method == 'POST':
        batch_name = request.form['batch_name'].strip()
        subject = request.form['subject'].strip()
        
        if not batch_name or not subject:
            flash("Batch Name and Subject are required!", "error")
            return redirect(url_for('register_batch'))
            
        teacher_id = session['user']['id']
        batch_id = create_batch(batch_name, subject, teacher_id)
        return redirect(url_for('register_students', batch_id=batch_id))
        
    return render_template('register_batch.html')

@app.route('/register_students/<int:batch_id>')
@login_required
def register_students(batch_id):
    batch = get_batch_by_id(batch_id)
    if not batch or str(batch['teacher_id']) != session['user']['id']:
        flash("Batch not found or unauthorized.", "error")
        return redirect(url_for('index'))
    return render_template('register_students.html', batch=batch)

@app.route('/registration_video_feed/<batch_name>/<subject>')
@login_required
def registration_video_feed(batch_name, subject):
    return Response(generate_registration_frames(batch_name, subject),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_student', methods=['POST'])
@login_required
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
@login_required
def capture_status(req_key):
    req = CAPTURE_REQUESTS.get(req_key)
    if not req:
        return jsonify({"error": "Not found"}), 404
    
    return jsonify({
        "count": req['count'],
        "done": req['done']
    })

@app.route('/complete_registration/<int:batch_id>', methods=['POST'])
@login_required
def complete_registration(batch_id):
    batch = get_batch_by_id(batch_id)
    if not batch or str(batch['teacher_id']) != session['user']['id']:
        flash("Unauthorized.", "error")
        return redirect(url_for('index'))

    if encode_batch_faces(batch_id, batch['batch_name'], batch['subject']):
        flash(f"Batch {batch['batch_name']} registered successfully!", "success")
    else:
        flash(f"Failed to encode faces. Were any students added?", "error")
    return redirect(url_for('registration_summary', batch_id=batch_id))

@app.route('/registration_summary/<int:batch_id>')
@login_required
def registration_summary(batch_id):
    batch = get_batch_by_id(batch_id)
    if not batch or str(batch['teacher_id']) != session['user']['id']:
        flash("Unauthorized.", "error")
        return redirect(url_for('index'))
    users = get_users_by_batch(batch_id)
    return render_template('registration_summary.html', batch=batch, count=len(users))

@app.route('/mark_attendance')
@login_required
def mark_attendance():
    teacher_id = session['user']['id']
    batches = get_batches(teacher_id)
    return render_template('select_batch.html', batches=batches)

@app.route('/attendance_viewfinder/<int:batch_id>')
@login_required
def attendance_viewfinder(batch_id):
    batch = get_batch_by_id(batch_id)
    if not batch or str(batch['teacher_id']) != session['user']['id']:
        flash("Unauthorized.", "error")
        return redirect(url_for('index'))
    return render_template('attendance_viewfinder.html', batch=batch)

@app.route('/attendance_video_feed/<int:batch_id>')
@login_required
def attendance_video_feed(batch_id):
    return Response(generate_face_recognition_frames(batch_id, app),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_last_event')
@login_required
def get_last_event():
    return jsonify(app.last_event)

@app.route('/attendance_summary/<int:batch_id>')
@login_required
def attendance_summary(batch_id):
    batch = get_batch_by_id(batch_id)
    if not batch or str(batch['teacher_id']) != session['user']['id']:
        flash("Unauthorized.", "error")
        return redirect(url_for('index'))
    summary = get_attendance_summary(batch_id)
    return render_template('attendance_summary.html', batch=batch, summary=summary)

if __name__ == '__main__':
    # Disable the reloader as it notoriously crashes heavy C-extensions (like dlib).
    # We MUST keep threaded=True so the video stream doesn't block AJAX requests.
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False, threaded=True)
