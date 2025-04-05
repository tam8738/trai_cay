import gradio as gr
import numpy as np
from PIL import Image
from keras.models import load_model
import tensorflow as tf

# Đảm bảo TensorFlow chỉ sử dụng CPU
tf.config.set_visible_devices([], 'GPU')

# Tải mô hình đã được huấn luyện
model = load_model('D:\XLA\trai_cay/model.h5')

# Các lớp hoa quả
classes = { 
    0: 'Apple',
    1: 'Banana',
    2: 'Grape',
    3: 'Mango',
    4: 'Strawberry',
}

# Hàm phân loại
def classify(img):
    image = Image.open(img)
    image = image.resize((100, 100))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    image = image / 255.0
    pred = model.predict(image)[0]
    sign = classes[np.argmax(pred)]
    print(sign)
    return sign

# Đường dẫn tới hình ảnh thử nghiệm
img = "./path_to_your_test_image.jpg"
classify(img)

# Giao diện Gradio
iface = gr.Interface(
    fn=classify, 
    inputs=gr.inputs.Image(shape=(100, 100)), 
    outputs=gr.outputs.Textbox()
)

iface.launch(share=True)
