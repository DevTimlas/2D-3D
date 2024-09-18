from flask import Flask, request, jsonify
from gradio_client import Client, handle_file

app = Flask(__name__)

# Gradio Client setup
client = Client("ohamidli/2D-3D", hf_token='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')

# Endpoint for human image prediction
@app.route('/predict_human', methods=['POST'])
def predict_human():
    file = request.files['image']
    file_path = f"./{file.filename}"
    file.save(file_path)
    result = client.predict(image=handle_file(file_path), api_name="/predict")
    return jsonify(result)

# Endpoint for garment image prediction
@app.route('/predict_garment', methods=['POST'])
def predict_garment():
    file = request.files['image']
    file_path = f"./{file.filename}"
    file.save(file_path)
    result = client.predict(image=handle_file(file_path), api_name="/predict_1")
    return jsonify(result)

# Endpoint for drape prediction (no file required)
@app.route('/predict_drape', methods=['GET'])
def predict_drape():
    result = client.predict(api_name="/predict_2")
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
