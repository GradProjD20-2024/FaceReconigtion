import os
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import face_recognition

app = Flask(__name__)

# Đường dẫn đến thư mục chứa hình ảnh
KNOWN_FACES_DIR = "known_faces"

# Hàm để load và chuyển đổi ảnh sang numpy array
def load_image(file_path):
    image = Image.open(file_path).convert("RGB")
    image_np = np.array(image)
    return image_np

# Hàm so sánh khuôn mặt từ ảnh tải lên với ảnh trong thư mục
def find_matching_face(uploaded_image_encoding):
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            known_image_path = os.path.join(KNOWN_FACES_DIR, filename)
            known_image_np = load_image(known_image_path)
            known_face_locations = face_recognition.face_locations(known_image_np)
            known_face_encodings = face_recognition.face_encodings(known_image_np, known_face_locations)

            if len(known_face_encodings) > 0:
                match = face_recognition.compare_faces([known_face_encodings[0]], uploaded_image_encoding)
                if match[0]:
                    return filename
    return None

# Route nhận ảnh và trả về tên file nếu khuôn mặt khớp
@app.route('/find_face', methods=['POST'])
def find_face():
    try:
        # Nhận ảnh từ request
        file = request.files['image']

        # Mở ảnh và chuyển đổi thành numpy array
        uploaded_image_np = load_image(file)

        # Tìm vị trí và encoding của khuôn mặt trong ảnh tải lên
        uploaded_face_locations = face_recognition.face_locations(uploaded_image_np)
        uploaded_face_encodings = face_recognition.face_encodings(uploaded_image_np, uploaded_face_locations)

        # Kiểm tra xem có khuôn mặt trong ảnh không
        if len(uploaded_face_encodings) == 0:
            return jsonify({
                "status_code": 400,
                "message": "Không tìm thấy khuôn mặt trong ảnh tải lên",
                "data": None
            }), 400

        # Tìm ảnh trùng khớp trong thư mục
        matching_filename = find_matching_face(uploaded_face_encodings[0])

        if matching_filename:
            return jsonify({
                "status_code": 200,
                "message": "Tìm thấy khuôn mặt trùng khớp",
                "data": {
                    "matching_file": matching_filename
                }
            }), 200
        else:
            return jsonify({
                "status_code": 404,
                "message": "Không tìm thấy khuôn mặt trùng khớp",
                "data": None
            }), 404

    except Exception as e:
        return jsonify({
            "status_code": 500,
            "message": "Lỗi hệ thống: " + str(e),
            "data": None
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
