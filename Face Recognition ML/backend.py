from flask import Flask, request, jsonify, send_from_directory
import os
import cv2
import face_recognition
import numpy as np
import MySQLdb

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Database connection configuration
db = MySQLdb.connect(host="feenix-mariadb.swin.edu.au",
                     user="s103844421",
                     passwd="181203",
                     db="s103844421_db")

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'message': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the uploaded image
        image = face_recognition.load_image_file(filepath)
        face_encodings = face_recognition.face_encodings(image)

        if len(face_encodings) == 0:
            return jsonify({'message': 'No face detected in the image'}), 400

        face_encoding = face_encodings[0]

        # Check for matches in the database
        cursor = db.cursor()
        cursor.execute("SELECT name, fpath, feature FROM face_recognition")
        records = cursor.fetchall()
        for record in records:
            db_name, db_fpath, db_feature = record
            db_feature = np.frombuffer(db_feature, dtype=np.float64)
            match = face_recognition.compare_faces([db_feature], face_encoding, tolerance=0.6)
            if match[0]:
                return handle_existing_person(db_name, db_fpath, filename, face_encoding, filepath)

        return handle_new_person(filename, face_encoding, filepath)

    return jsonify({'message': 'Failed to upload image'}), 400

def handle_existing_person(db_name, db_fpath, filename, face_encoding, filepath):
    similarity = face_recognition.face_distance([db_fpath], face_encoding)
    if similarity < 0.2:  # Adjust the threshold as necessary
        return jsonify({'message': f'Confirm that this is {db_name}.', 'buttons': ['YES', 'NO']})
    return jsonify({'message': f'Person {db_name} already exists in the database. Do you want to add another image for {db_name}?', 'buttons': ['YES', 'NO']})

def handle_new_person(filename, face_encoding, filepath):
    return jsonify({'message': 'Enter the name of this person:', 'buttons': ['Submit']})

@app.route('/add_person', methods=['POST'])
def add_person():
    data = request.json
    name = data['name']
    filepath = data['filepath']
    face_encoding = np.asarray(data['face_encoding'])

    cursor = db.cursor()
    cursor.execute("SELECT name FROM face_recognition WHERE name=%s", (name,))
    record = cursor.fetchone()
    if record:
        return jsonify({'message': f'Person {name} already exists in the database. Please use a different name.'}), 400

    cursor.execute("INSERT INTO face_recognition (name, fpath, feature) VALUES (%s, %s, %s)",
                   (name, filepath, face_encoding.tobytes()))
    db.commit()

    return jsonify({'message': f'Successfully added person {name} into the database.'})

@app.route('/confirm_person', methods=['POST'])
def confirm_person():
    data = request.json
    name = data['name']
    filepath = data['filepath']
    face_encoding = np.asarray(data['face_encoding'])

    cursor = db.cursor()
    cursor.execute("INSERT INTO face_recognition (name, fpath, feature) VALUES (%s, %s, %s)",
                   (name, filepath, face_encoding.tobytes()))
    db.commit()

    return jsonify({'message': 'Face recognition successful.'})

if __name__ == '__main__':
    app.run(debug=True)
