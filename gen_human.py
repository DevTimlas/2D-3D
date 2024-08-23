import os
import torch
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, send_file
from io import BytesIO

# Set up paths
image_dir = "/Users/mac/fivv/2D-3D/latest/pifuhd/sample_images"
output_dir = "/Users/mac/fivv/2D-3D/latest/pifuhd/results/pifuhd_final/recon"

import os

os.chdir('/Users/mac/fivv/2D-3D/latest/lightweight-human-pose-estimation.pytorch')
print('cwd', os.getcwd())

import sys
sys.path.append('/Users/mac/fivv/2D-3D/latest/lightweight-human-pose-estimation.pytorch')
# Import necessary modules from lightweight human pose estimation
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose
import demo

# Initialize Flask app
app = Flask(__name__)

# Load pose estimation model
net = PoseEstimationWithMobileNet()
checkpoint = torch.load('/Users/mac/fivv/2D-3D/latest/lightweight-human-pose-estimation.pytorch/checkpoint_iter_370000.pth', map_location='cpu')
load_state(net, checkpoint)
net = net.eval()

# Preprocess function to crop image
def get_rect(net, image_path, height_size=512):
    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    orig_img = img.copy()
    heatmaps, pafs, scale, pad = demo.infer_fast(net, img, height_size, stride, upsample_ratio, cpu=True)

    total_keypoints_num = 0
    all_keypoints_by_type = []
    for kpt_idx in range(num_keypoints):
        total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
    for kpt_id in range(all_keypoints.shape[0]):
        all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
        all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale

    rects = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
        valid_keypoints = []
        for kpt_id in range(num_keypoints):
            if pose_entries[n][kpt_id] != -1.0:
                pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
                valid_keypoints.append([pose_keypoints[kpt_id, 0], pose_keypoints[kpt_id, 1]])
        valid_keypoints = np.array(valid_keypoints)

        if pose_entries[n][10] != -1.0 or pose_entries[n][13] != -1.0:
            pmin = valid_keypoints.min(0)
            pmax = valid_keypoints.max(0)
            center = (0.5 * (pmax[:2] + pmin[:2])).astype(np.int32)
            radius = int(0.65 * max(pmax[0] - pmin[0], pmax[1] - pmin[1]))
        elif pose_entries[n][10] == -1.0 and pose_entries[n][13] == -1.0 and pose_entries[n][8] != -1.0 and pose_entries[n][11] != -1.0:
            center = (0.5 * (pose_keypoints[8] + pose_keypoints[11])).astype(np.int32)
            radius = int(1.45 * np.sqrt(((center[None, :] - valid_keypoints) ** 2).sum(1)).max(0))
            center[1] += int(0.05 * radius)
        else:
            center = np.array([img.shape[1] // 2, img.shape[0] // 2])
            radius = max(img.shape[1] // 2, img.shape[0] // 2)

        x1 = center[0] - radius
        y1 = center[1] - radius

        rects.append([x1, y1, 2 * radius, 2 * radius])

    return np.array(rects)

# Function to process image and generate mesh
def process_image(image):
    # Save the uploaded image
    image_path = os.path.join(image_dir, "uploaded_image.png")
    image.save(image_path)

    # Preprocess the image to get rect
    rects = get_rect(net, image_path)
    rect_path = image_path.replace('.png', '_rect.txt')
    np.savetxt(rect_path, rects, fmt='%d')

    # Run PIFuHD
    os.chdir('/Users/mac/fivv/2D-3D/latest/pifuhd/')
    os.system(f"python -m apps.simple_test -r 256 --use_rect -i {image_dir}")

    # Get the result object file path
    file_name = "uploaded_image"
    obj_path = os.path.join(output_dir, f"result_{file_name}_256.obj")

    return obj_path

# Flask route to process the image and return the OBJ file
@app.route('/generate-mesh', methods=['POST'])
def generate_mesh():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file:
        # Convert the file to an image and process it
        image = Image.open(file.stream)
        obj_path = process_image(image)
        
        # Return the OBJ file
        return send_file(obj_path, as_attachment=True)

# Run the Flask app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6000)
