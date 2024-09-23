from flask import Flask, request, send_file
from gradio_client import Client, handle_file
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Gradio Client setup
client = Client("ohamidli/2D-3D", hf_token='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')

# Endpoint for human image prediction and .obj download
@app.route('/predict_human', methods=['POST'])
def predict_human():
    file = request.files['image']
    file_path = f"./{file.filename}"
    file.save(file_path)
    result_path = client.predict(image=handle_file(file_path), api_name="/predict")

    # Assuming result_path is the path to the .obj file, return the actual file
    result_path = result_path.strip().strip('"')  # clean up the path
    return send_file(result_path, as_attachment=True)

# Endpoint for garment image prediction and .obj download
@app.route('/predict_garment', methods=['POST'])
def predict_garment():
    file = request.files['image']
    file_path = f"./{file.filename}"
    file.save(file_path)
    result_path = client.predict(image=handle_file(file_path), api_name="/predict_1")

    # Assuming result_path is the path to the .obj file, return the actual file
    result_path = result_path.strip().strip('"')  # clean up the path
    return send_file(result_path, as_attachment=True)

# Endpoint for drape prediction and .obj download
@app.route('/predict_drape', methods=['GET'])
def predict_drape():
    result_path = client.predict(api_name="/predict_2")

    # Assuming result_path is the path to the .obj file, return the actual file
    result_path = result_path.strip().strip('"')  # clean up the path
    return send_file(result_path, as_attachment=True)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)
