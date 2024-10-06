from flask import Flask, request, send_file
from gradio_client import Client, handle_file
import os
import time  # For handling time calculations
from flask_cors import CORS
from werkzeug.utils import secure_filename
from rich import print

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './uploads'
RESULT_FOLDER = './results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

human_obj_file = None
garment_obj_file = None
last_gpu_error_time = None  # To track the last time a GPU quota error occurred
cooldown_period = 5 * 60  # 5 minutes cooldown period in seconds

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Gradio Client setup
client = Client("ohamidli/2D-3D", hf_token='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')

def is_within_gpu_quota_cooldown():
    """Check if a GPU quota error happened within the last 5 minutes."""
    global last_gpu_error_time
    if last_gpu_error_time is None:
        return False  # No GPU quota error has occurred yet
    return time.time() - last_gpu_error_time < cooldown_period

def update_gpu_error_time():
    """Update the time of the last GPU quota error."""
    global last_gpu_error_time
    last_gpu_error_time = time.time()

# Endpoint for human image prediction and .obj download
@app.route('/predict_human', methods=['POST'])
def predict_human():
    global human_obj_file

    if is_within_gpu_quota_cooldown():
        return "GPU quota exceeded. Please wait a few minutes before making another request.", 429

    print("Received request!")
    if 'image' not in request.files:
        return "No file part", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400
    if not allowed_file(file.filename):
        return "Invalid file type. Only PNG, JPG, WEBP, and JPEG are allowed.", 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        # Get the result from the model
        human_obj_file = client.predict(image=handle_file(file_path), api_name="/predict").strip().strip('"')
        return send_file(human_obj_file, as_attachment=True)
    except Exception as e:
        if "exceeded your GPU quota" in str(e):
            update_gpu_error_time()  # Update the time if GPU quota error occurs
            return "GPU quota exceeded. Please try again in a few minutes.", 429
        else:
            print(f"Error during prediction: {str(e)}")
            return "Prediction failed due to an internal error.", 500

# Endpoint for garment image prediction and .obj download
@app.route('/predict_garment', methods=['POST'])
def predict_garment():
    global garment_obj_file

    if is_within_gpu_quota_cooldown():
        return "GPU quota exceeded. Please wait a few minutes before making another request.", 429

    if 'image' not in request.files:
        return "No file part", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400
    if not allowed_file(file.filename):
        return "Invalid file type. Only PNG, JPG, WEBP and JPEG are allowed.", 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        # Get the result from the model
        garment_obj_file = client.predict(image=handle_file(file_path), api_name="/predict_1").strip().strip('"')
        return send_file(garment_obj_file, as_attachment=True)
    except Exception as e:
        if "exceeded your GPU quota" in str(e):
            update_gpu_error_time()  # Update the time if GPU quota error occurs
            return "GPU quota exceeded. Please try again in a few minutes.", 429
        else:
            print(f"Error during prediction: {str(e)}")
            return "Prediction failed due to an internal error.", 500

# Endpoint for drape prediction and .obj download
@app.route('/predict_drape', methods=['GET'])
def predict_drape():
    if human_obj_file is None:
        return "Human object file not generated. Please generate it first.", 400
    if garment_obj_file is None:
        return "Garment object file not generated. Please generate it first.", 400

    if is_within_gpu_quota_cooldown():
        return "GPU quota exceeded. Please wait a few minutes before making another request.", 429

    try:
        result_path = client.predict(api_name="/predict_2").strip().strip('"') # 
        return send_file(result_path, as_attachment=True)
    except Exception as e:
        if "exceeded your GPU quota" in str(e):
            update_gpu_error_time()  # Update the time if GPU quota error occurs
            return "GPU quota exceeded. Please try again in a few minutes.", 429
        else:
            print(f"Error during prediction: {str(e)}")
            return "Failed to drape the models", 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)
