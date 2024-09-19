import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2


# model = tf.keras.models.load_model('CNN/tumor_detection/results/model/cnn_tumor.h5')

def make_prediction(img,model):
    # img=cv2.imread(img)
    img=Image.fromarray(img)
    img=img.resize((128,128))
    img=np.array(img)
    input_img = np.expand_dims(img, axis=0)
    res = model.predict(input_img)
    if res:
        print("Tumor Detected")
    else:
        print("No Tumor")
    return res
        
make_prediction(cv2.imread("deeplearning/CNN/tumordata/yes/y964.jpg"),model)
print("--------------------------------------\n")
make_prediction(cv2.imread("deeplearning/CNN/tumordata/no/no978.jpg"),model)
print("--------------------------------------\n")


st.title("Tumor Detection using CNN")


uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    resized_image = image.resize((300, 200))
    st.image(resized_image, caption="Uploaded Image")
    
    processed_image = preprocess_image(image)
    result = predict_tumor(processed_image)
    st.info(f"Prediction: {result}")
