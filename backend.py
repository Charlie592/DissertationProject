# backend.py adjustment
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from main import process_file

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = '/Users/charlierobinson/Documents/Code/DissertationCode/Project 2/uploads'

@app.route('/api/data', methods=['GET', 'POST'])
def handle_data():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"message": "No file part"}), 400
        
        file = request.files['file']
        impute_missing_values = request.form.get('imputeMissingValues') == 'true'  # Convert string to boolean
        
        if file.filename == '':
            return jsonify({"message": "No selected file"}), 400
        
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        
        process_file(save_path, save_path, impute_missing_values)
        
        return jsonify({"message": "File uploaded and processed successfully"}), 200

if __name__ == '__main__':
    app.run(debug=True)
