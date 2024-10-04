from flask import Flask, request, send_file
from gradio_client import Client, handle_file
import os
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Gradio Client setup
client = Client("ohamidli/2D-3D", hf_token='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')

# Endpoint for human image prediction and .obj download
@app.route('/predict_human', methods=['POST'])
def predict_human():
    print("Received request!")
    if 'image' not in request.files:
        return "No file part", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400
    if not allowed_file(file.filename):
        return "Invalid file type. Only PNG, JPG, and JPEG are allowed.", 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    result_path = client.predict(image=handle_file(file_path), api_name="/predict")
    result_path = result_path.strip().strip('"')  # Clean up the path
    return send_file(result_path, as_attachment=True)

# Endpoint for garment image prediction and .obj download
@app.route('/predict_garment', methods=['POST'])
def predict_garment():
    if 'image' not in request.files:
        return "No file part", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400
    if not allowed_file(file.filename):
        return "Invalid file type. Only PNG, JPG, and JPEG are allowed.", 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    result_path = client.predict(image=handle_file(file_path), api_name="/predict_1")
    result_path = result_path.strip().strip('"')  # Clean up the path
    return send_file(result_path, as_attachment=True)

# Endpoint for drape prediction and .obj download
@app.route('/predict_drape', methods=['GET'])
def predict_drape():
    result_path = client.predict(api_name="/predict_2")
    result_path = result_path.strip().strip('"')  # Clean up the path
    return send_file(result_path, as_attachment=True)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)
