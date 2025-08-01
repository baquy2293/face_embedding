import os
from flask import Flask, request, jsonify
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import io
import torch
import numpy as np

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=123, margin=20, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

from torchvision import transforms

def extract_face_vector(file, save_path=None):
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    face = mtcnn(img)
    if face is None:
        return None

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Đảm bảo face đúng chuẩn ToPILImage
        face_cpu = face.detach().cpu()  # Tensor shape: (3, 160, 160), dtype: float32, range [0,1]
        to_pil = transforms.ToPILImage()
        face_pil = to_pil(face_cpu)
        face_pil.save(save_path)

    face = face.unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(face)
    return embedding.squeeze().cpu().numpy()


def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return float(sim)

@app.route('/compare-faces', methods=['POST'])
def compare_faces():
    if 'face1' not in request.files or 'face2' not in request.files:
        return jsonify({"error": "Thiếu file face1 hoặc face2"}), 400

    try:
        face1_file = request.files['face1']
        face2_file = request.files['face2']

        # Tạo thư mục faces và lưu ảnh cắt
        vec1 = extract_face_vector(face1_file, save_path='faces/crop_face1.jpg')
        face2_file.seek(0)
        vec2 = extract_face_vector(face2_file, save_path='faces/crop_face2.jpg')

        if vec1 is None or vec2 is None:
            return jsonify({"error": "Không tìm thấy khuôn mặt trong 1 hoặc 2 ảnh"}), 400

        similarity = cosine_similarity(vec1, vec2)
        match = similarity > 0.7

        return jsonify({
            "similarity": similarity,
            "match": match,
            "message": "Giống nhau" if match else "Không giống nhau"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=True)
