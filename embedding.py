from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from mtcnn import MTCNN
import cv2
import os
import uuid

# ---- Load model và detector ----
MODEL_PATH = "facenet.tflite"   # Hoặc "facenet.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
input_shape = input_details[0]['shape']
detector = MTCNN()

# ---- Hàm detect & crop & lưu mặt ----
def detect_and_crop_face_mtcnn(image_bytes, desired_size=112, save_path=None, suffix=""):
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = detector.detect_faces(img_rgb)
    if faces:
        x, y, w, h = faces[0]['box']
        x, y = max(x, 0), max(y, 0)
        face_img = img_rgb[y:y+h, x:x+w]
    else:
        face_img = img_rgb

    face_img = cv2.resize(face_img, (desired_size, desired_size))
    pil_img = Image.fromarray(face_img)

    # Lưu lại ảnh nếu truyền save_path
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename = f"face_{uuid.uuid4().hex[:8]}{suffix}.jpg"
        pil_img.save(os.path.join(save_path, filename))

    return pil_img

def preprocess_image_bytes(image_bytes, input_shape, save_path=None, suffix=""):
    face_pil = detect_and_crop_face_mtcnn(
        image_bytes,
        desired_size=input_shape[1],
        save_path=save_path,
        suffix=suffix
    )
    img = np.asarray(face_pil).astype("float32")
    img = (img - 127.5) / 128.0
    img = np.expand_dims(img, axis=0)
    return img

def get_embedding(interpreter, img):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    emb = interpreter.get_tensor(output_details[0]['index'])
    return emb[0]

def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))

# ---- Flask API ----
app = Flask(__name__)

@app.route('/face-similarity', methods=['POST'])
def face_similarity():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Hãy gửi cả hai file ảnh (image1, image2)'}), 400

    image1 = request.files['image1'].read()
    image2 = request.files['image2'].read()

    try:
        # Tự động crop và lưu ảnh crop
        img1 = preprocess_image_bytes(image1, input_shape, save_path="cropped_faces", suffix="_1")
        img2 = preprocess_image_bytes(image2, input_shape, save_path="cropped_faces", suffix="_2")

        emb1 = get_embedding(interpreter, img1)
        emb2 = get_embedding(interpreter, img2)

        sim = cosine_similarity(emb1, emb2)
        return jsonify({
            'cosine_similarity': round(sim, 4),
            'same_person': sim > 0.5
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
