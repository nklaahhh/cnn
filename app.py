import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import subprocess
import streamlit as st

def upgrade_pip():
    with st.spinner("Upgrading pip..."):
        # Upgrade pip using subprocess to run the command
        result = subprocess.run(["python", "-m", "pip", "install", "--upgrade", "pip"], capture_output=True, text=True)
        
        # Check for errors in the result
        if result.returncode == 0:
            st.success("Pip has been upgraded successfully!")
        else:
            st.error(f"Failed to upgrade pip:\n{result.stderr}")

# Button in Streamlit to trigger the pip upgrade
if st.button("Upgrade pip"):
    upgrade_pip()




def load_model():
    model = tf.keras.models.load_model('cnn_tumor2.h5')
    return model

model = load_model()


def preprocess_image(image):
    image = image.resize((128,128)) 
    image = np.array(image) / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image


def predict_tumor(image):
    predictions = model.predict(image)
    if predictions[0] > 0.5:  
        return "Tumorous"
    else:
        return "Non-tumorous"


st.title("Tumor Detection using CNN")


uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    resized_image = image.resize((300, 200))
    st.image(resized_image, caption="Uploaded Image")
    
    processed_image = preprocess_image(image)
    result = predict_tumor(processed_image)
    st.info(f"Prediction: {result}")
