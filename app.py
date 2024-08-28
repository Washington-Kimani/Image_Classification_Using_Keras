import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load your model (ensure your model is accessible in the same directory or provide the correct path)
model = load_model('/home/codename/projects/ds n ml/models/Image_Classification_Using_Keras/cat_dog.keras')

st.title('Cat vs Dog Classifier')

# Initial text to confirm the page is loading
st.write("Welcome to the Cat vs Dog Classifier!")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Check if the file uploader is working
if uploaded_file is not None:
    st.write("File uploaded successfully!")

    # Open the uploaded image file
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Image displayed successfully!")

    # Preprocess the image
    try:
        image = image.resize((100, 100))  # Resize to match model input size
        image = np.array(image)           # Convert image to array
        image = image / 255.0             # Normalize the image
        image = image.reshape(1, 100, 100, 3)  # Reshape for the model

        st.write("Image preprocessing successful!")
        
        # Make the prediction
        y_pred = model.predict(image)
        y_pred = y_pred > 0.5
        
        # Interpret the prediction
        if y_pred == 0:
            pred = 'dog'
        else:
            pred = 'cat'
        
        st.write(f"Our model says it is a: **{pred}**")
    except Exception as e:
        st.write("Error during image processing or prediction:")
        st.write(e)
else:
    st.write("Please upload an image file.")
