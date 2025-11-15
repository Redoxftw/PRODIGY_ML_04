# --- Task 4: CNN Trainer (THE *FINAL* CORRECT VERSION) ---
#
# I've been a fool.
#
# The Bug: I never "normalized" the images. I fed raw [0, 255] pixels
# to a model that expects [-1, 1] pixels.
# The model trained on garbage, so it's predicting garbage.
#
# The Fix:
# 1. I will add the *correct* MobileNetV2 pre-processing step
#    (which scales pixels to [-1, 1]) to my training data.
# 2. I will re-train the model.
# 3. I will use this *same* pre-processing in the app.
#
# THIS IS IT.

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from keras.applications import MobileNetV2
# THIS IS THE NEW IMPORT
from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import image_dataset_from_directory
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

print("--- Task 4: CNN Trainer (THE *FINAL* CORRECT VERSION) ---")
print(f"Using TensorFlow version: {tf.__version__}")

# --- 1. Setup ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'
MODEL_FILE = 'hand_gesture_model_FINAL.h5' # New name for the new model

# --- 2. Load Data ---
print(f"Loading MASSIVE training data from {TRAIN_DIR}...")
train_ds = image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    color_mode='grayscale' # Load as grayscale
)

print(f"Loading test data from {TEST_DIR}...")
test_ds = image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    color_mode='grayscale'
)

class_names = train_ds.class_names
print(f"Found {len(class_names)} classes: {class_names}")

# --- 3. Pre-process the Data ---

# This function will now do EVERYTHING:
# 1. Stack grayscale to 3-channel
# 2. Apply the MobileNetV2 normalization (pixels from [0, 255] -> [-1, 1])
def adapt_and_preprocess(image, label):
    image = tf.image.grayscale_to_rgb(image)
    image = preprocess_input(image) # THIS IS THE CRITICAL NEW LINE
    return image, label

train_ds = train_ds.map(adapt_and_preprocess)
test_ds = test_ds.map(adapt_and_preprocess)

# Cache for speed
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
print("Data pre-processing complete (Grayscale -> 3-Channel -> Normalized [-1, 1]).")

# --- 4. Build the Model (Same as before) ---
print("Building model with MobileNetV2 base...")
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs = Input(shape=(224, 224, 3))
# We pass the inputs through the base_model
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(len(class_names), activation='softmax')(x)
model = Model(inputs, outputs)

# --- 5. Compile the Model (Same as before) ---
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Model built and compiled successfully.")
model.summary()

# --- 6. Train the Model (This will be fast now!) ---
print("\n--- Starting Model Training (THE *FINAL* TIME) ---")
print("This should be much faster and more accurate...")

EPOCHS = 5
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS
)

print("--- Training Complete ---")

# --- 7. Evaluate the Model ---
print("Evaluating model on the *test* dataset (Person 01)...")
loss, accuracy = model.evaluate(test_ds)

print(f"\n--- Model Evaluation ---")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")

# --- 8. Save the Model ---
model.save(MODEL_FILE)
print(f"\nModel saved successfully to '{MODEL_FILE}'")
print("\n--- Task 4 Trainer Script Finished ---")
print("You can now run 'streamlit run app.py' to use this new, FINAL model.")