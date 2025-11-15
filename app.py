# --- Task 4: Hand Gesture Recognition App (THE *FINAL* CORRECT VERSION) ---
#
# The Problem: My pre-processing in the app and trainer were DIFFERENT.
#
# The Fix:
# 1. I re-trained my model using the *correct* MobileNetV2 normalization
#    (scaling pixels to [-1, 1]).
# 2. This app will now use that *exact same* normalization.
#
# This means the app and the trainer are finally speaking the same language.
# This must be it.

import streamlit as st
import tensorflow as tf
from tensorflow import keras
# THIS IS THE NEW, CRITICAL IMPORT
from keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import cv2  # OpenCV for image processing
from PIL import Image # Pillow for opening the image file
import os

# --- 1. Setup & Model Loading ---

# Load the NEWEST, FINAL model
MODEL_FILE = 'hand_gesture_model_FINAL.h5'
FRIENDLY_NAMES = [
    "Palm", "L-Shape", "Fist", "Fist (Moved)", "Thumb Up",
    "Index", "OK", "Palm (Moved)", "C-Shape", "Down"
]
IMG_SIZE = (224, 224)

@st.cache_resource
def load_my_model(model_path):
    """Loads my trained .h5 model."""
    if os.path.exists(model_path):
        model = keras.models.load_model(model_path)
        print("Final model loaded successfully!")
        return model
    else:
        st.error(f"Error: Model file '{model_path}' not found!")
        st.write("Please run the new 'task_04.py' first to train and save the model.")
        st.stop()

# --- 2. Image Processing Function (THE *FINAL* CORRECTED VERSION) ---
def process_image(image_file):
    """Takes the webcam snapshot and prepares it for the model."""
    
    # 1. Open the image using PIL
    img = Image.open(image_file)
    
    # 2. Convert to NumPy array
    img_array = np.array(img)
    
    # 3. Use OpenCV to convert to Grayscale
    gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # 4. Resize to 224x224
    resized_img = cv2.resize(gray_img, IMG_SIZE)
    
    # 5. Stack channels (to "fake" RGB for the model)
    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)
    
    # 6. Expand dimensions
    # Change shape from (224, 224, 3) to (1, 224, 224, 3)
    img_batch = np.expand_dims(rgb_img, axis=0)
    
    # 7. NORMALIZE (THE *CORRECT* WAY)
    # This applies the MobileNetV2 scaling (pixels -> [-1, 1])
    # This is the *exact* same function used in the trainer.
    processed_batch = preprocess_input(img_batch)
    
    # I'll return the grayscale image just to show what we're looking at
    return processed_batch, resized_img

# --- 3. Build the Streamlit App ---

st.title("✋ Hand Gesture Recognition (Final Attempt)")
st.write("This CNN model was trained with the *correct* normalization (pixels from [-1, 1]). This is the one.")

# Load the model
model = load_my_model(MODEL_FILE)

if model:
    st.header("Human-Computer Interaction")
    st.write("Take a picture of your hand.")
    
    picture = st.camera_input("Take a snapshot:")

    if picture is not None:
        
        # 1. Show the original
        st.image(picture, caption="Here's the original snapshot:", use_container_width=True)
        
        with st.spinner("Analyzing... (for real this time)..."):
            
            # 2. Process the image (Correctly this time)
            processed_img_batch, gray_image_to_show = process_image(picture)
            
            # 3. Make the prediction!
            prediction = model.predict(processed_img_batch)
            
            # 4. Find the winning class
            predicted_index = np.argmax(prediction)
            predicted_class_name = FRIENDLY_NAMES[predicted_index]
            confidence = np.max(prediction) * 100
        
        # 5. Show the result!
        st.success(f"I think that's a: **{predicted_class_name}**")
        st.write(f"I'm **{confidence:.2f}%** sure.")
        
        # 6. Show what the model *actually* saw (the pre-normalized version)
        st.write("---")
        st.header("What the Model *Actually* Saw:")
        st.image(gray_image_to_show, caption="The resized, grayscale image (before normalization).", use_container_width=True)