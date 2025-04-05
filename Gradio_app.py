import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Bắt buộc TensorFlow sử dụng CPU

import gradio as gr
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Tải mô hình đã được huấn luyện
model_path = '/content/drive/MyDrive/trai_cay/model.h5'  # Thay thế bằng đường dẫn chính xác đến mô hình của bạn
model = load_model(model_path)

# Dictionary để gán nhãn cho các lớp hoa quả
classes = {
    0: 'Apple',
    1: 'Banana',
    2: 'Grape',
    3: 'Mango',
    4: 'Strawberry',
}

def classify_image(img):
    # Xử lý hình ảnh
    image = img.resize((1000, 100))
    image = np.expand_dims(np.array(image) / 255.0, axis=0)
    
    # Dự đoán và cập nhật nhãn
    pred = np.argmax(model.predict(image), axis=-1)[0]
    return classes[pred]

# Tạo giao diện Gradio
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Text(),
    title="Fruit Classification",
    description="Tải lên một hình ảnh của trái cây và nhận diện loại trái cây."
)

# Chạy giao diện
iface.launch()