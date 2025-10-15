# Face-Mask-Detection
3-Class Face Mask Detection
A deep learning model to classify if a person is wearing a mask correctly, incorrectly, or not at all. Built with TensorFlow/Keras using transfer learning on the MobileNetV2 architecture.

✨ Key Features
Multi-Class Classification: Detects three states: with_mask, without_mask, and mask_weared_incorrect.

Transfer Learning: Uses pre-trained MobileNetV2 for high accuracy and fast training.

Robust Training: Prevents overfitting using Data Augmentation, Dropout, and Early Stopping.

🚀 Usage
Download the 3class_face_mask_detector.h5 file and use the following snippet to make predictions.

Python

import tensorflow as tf
import numpy as np

# Load the model
model = tf.keras.models.load_model('3class_face_mask_detector.h5')

# Define class names (ensure order is correct)
class_names = ['mask_weared_incorrect', 'with_mask', 'without_mask']

# Load and preprocess your image (must be 224x224)
# image = ... # your preprocessed image as a numpy array
# image_batch = np.expand_dims(image, axis=0)

# Make a prediction
# prediction = model.predict(image_batch)
# predicted_class = class_names[np.argmax(prediction)]
# print(f"Prediction: {predicted_class}")
🛠️ Technologies Used
TensorFlow / Keras

Scikit-learn

Numpy

Matplotlib / Seaborn
