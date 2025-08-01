from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import time
import numpy as np
import torch.nn.functional as F

# Ép dùng CPU hoặc dùng GPU nếu có
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Khởi tạo detector và model embedding
mtcnn = MTCNN(image_size=160, margin=0, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_face_embedding(img_path):
    """ Trả về embedding 512-d từ ảnh """
    img = Image.open(img_path)
    face = mtcnn(img)
    if face is not None:
        face = face.unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(face)
        return embedding.squeeze(0)
    else:
        print(f"❌ Không tìm thấy khuôn mặt trong ảnh: {img_path}")
        return None

def cosine_similarity(emb1, emb2):
    """ Tính độ tương đồng cosine giữa 2 embedding """
    return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

def compare_faces(img_path1, img_path2, threshold=0.6):
    """ So sánh 2 ảnh khuôn mặt """
    start = time.time()

    emb1 = get_face_embedding(img_path1)
    emb2 = get_face_embedding(img_path2)

    if emb1 is None or emb2 is None:
        return

    similarity = cosine_similarity(emb1, emb2)

    print(f"🧠 Cosine similarity: {similarity:.4f}")
    if similarity > threshold:
        print("✅ Hai ảnh giống mặt (cùng người)")
    else:
        print("❌ Hai ảnh khác mặt (khác người)")

    print(f"⏱️ Thời gian so sánh: {time.time() - start:.4f} giây")

# 👉 Gọi hàm với 2 ảnh bạn muốn so sánh
compare_faces("./image/abc.jpg", "./image/IMG_7186.JPG")
