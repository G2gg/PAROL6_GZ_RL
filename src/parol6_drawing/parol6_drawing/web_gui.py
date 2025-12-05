#!/usr/bin/env python3

import os
import threading
import subprocess
import time
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from ament_index_python.resources import get_resource

_, package_share_path = get_resource('packages', 'parol6_gui')
templates_path = os.path.join(package_share_path, 'share', 'parol6_gui', 'templates')

app = Flask(__name__)
app.secret_key = 'robot_drawing_app_secret_key'
app.template_folder = os.path.join(package_share_path, 'share', 'parol6_gui', 'templates')

# Configuration
UPLOAD_FOLDER = '/tmp/robot_draw_svg'
ALLOWED_EXTENSIONS = {'svg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables
drawing_process = None
is_drawing = False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    global is_drawing
    return render_template('svg_index.html', is_drawing=is_drawing)

@app.route('/upload', methods=['POST'])
def upload_file():
    global is_drawing
    
    if is_drawing:
        flash('Robot is currently drawing. Please wait until it finishes.')
        return redirect(url_for('index'))
    
    # Check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    # If user does not select file, browser also submits an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get drawing plane selection
        drawing_plane = request.form.get('drawing_plane', 'XY')
        
        # Start drawing in a separate thread
        threading.Thread(target=start_drawing, args=(filepath, drawing_plane)).start()
        
        flash(f'File {filename} uploaded and robot is drawing now!')
        return redirect(url_for('index'))
    
    flash('Invalid file type. Please upload an SVG file.')
    return redirect(url_for('index'))

@app.route('/status')
def status():
    global is_drawing
    return {'is_drawing': is_drawing}

@app.route('/stop', methods=['POST'])
def stop_drawing():
    global drawing_process, is_drawing
    
    if drawing_process and is_drawing:
        # Send SIGINT to the process
        drawing_process.terminate()
        drawing_process = None
        is_drawing = False
        flash('Drawing stopped')
    else:
        flash('No active drawing to stop')
    
    return redirect(url_for('index'))

def start_drawing(svg_file, drawing_plane):
    global drawing_process, is_drawing
    
    try:
        is_drawing = True
        
        # Build the command
        cmd = [
            'ros2', 'run', 'parol6_drawing', 'robot_draw_using_gui', 
            '--ros-args',
            '-p', f'svg_file:={svg_file}',
            '-p', f'draw_plane:={drawing_plane}'
        ]
        
        # Execute the command
        drawing_process = subprocess.Popen(cmd)
        
        # Wait for the process to complete
        drawing_process.wait()
        
    except Exception as e:
        print(f"Error starting drawing process: {e}")
    finally:
        drawing_process = None
        is_drawing = False


def main(args=None):
    """Entry point for the ROS2 node"""
    app.run(host='0.0.0.0', port=5000, debug=True)

# Then modify the if __name__ == '__main__' block:
if __name__ == '__main__':
    main()