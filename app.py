import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import brain_tumor
import pneumonia
from util import set_background, set_sidebar_background

# Image preprocessing constants
img_width, img_height = 150, 150
threshold = 0.7
class_names = ['Brain MRI', 'Chest X-ray', 'Invalid Image']

# Load the model
initial_model_path = "C:\\Users\\sachi\\OneDrive\\Desktop\\brain tumor\\stark\\brain_chest_invalid_model.h5"
initial_model = load_model(initial_model_path)

def classify_image(img):
    img = img.resize((img_width, img_height))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = initial_model.predict(img_array)
    max_prob = np.max(predictions)

    if max_prob < threshold:
        return "Error: Uncertain classification", max_prob, max_prob

    class_idx = np.argmax(predictions)
    return class_names[class_idx], max_prob, max_prob

def login():
    placeholder = st.empty()
    actual_user_name = "user"
    actual_password = "12345"

    with placeholder.form("login"):
        st.markdown("<h3 style='color: black;'>Enter your credentials</h3>", unsafe_allow_html=True)
        user_name = st.text_input("User Name")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

    if submit and user_name == actual_user_name and password == actual_password:
        st.session_state["logged_in"] = True
        placeholder.empty()
    elif submit:
        st.error("Login failed")

st.set_page_config(
    page_title="Medical Image Analysis",
    page_icon="👨‍⚕️",
)

# Set background images for the page and sidebar
set_background('C:\\Users\\sachi\\OneDrive\\Desktop\\brain tumor\\stark\\background.jpg')
set_sidebar_background('C:\\Users\\sachi\\OneDrive\\Desktop\\brain tumor\\stark\\background.jpg')

# Title for the application
st.markdown("<h1 style='color: black;'>🩺 Welcome to our Medical Image Analysis website</h1>", unsafe_allow_html=True)

# Login process
if "logged_in" not in st.session_state:
    login()

# Main application after login
if "logged_in" in st.session_state and st.session_state["logged_in"]:
    st.sidebar.markdown("<h2 style='color: red;'>🧬 Select Disease Detection Option</h2>", unsafe_allow_html=True)

    # Sidebar dropdown to choose between Brain Tumor and Pneumonia
    disease_option = st.sidebar.selectbox("Choose Disease Type", ["Brain Tumor", "Pneumonia"])

    # Display instructions and file uploader based on disease selection
    st.markdown(f"<h2 style='color: black;'>🔬 Provide a {'Brain MRI' if disease_option == 'Brain Tumor' else 'Chest X-ray'} image for analysis 🔍</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    submit = st.button("Predict")

    if submit and uploaded_file:
        image = Image.open(uploaded_file)

        # Based on the selected disease, call the corresponding prediction function
        if disease_option == "Brain Tumor":
            brain_tumor.prediction(image)
        elif disease_option == "Pneumonia":
            pneumonia.prediction(image)
        else:
            st.error("Invalid option selected. Please try again.")
