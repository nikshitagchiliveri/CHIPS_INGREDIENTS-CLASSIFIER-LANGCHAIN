import streamlit as st
import pandas as pd
import joblib
import easyocr
import numpy as np
from PIL import Image
import os

# Custom CSS for improved UI styling
st.markdown("""
    <style>
    .container {
        max-width: 700px;
        margin: auto;
        padding: 2em;
        background-color: #f9f9f9;
        border-radius: 8px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
    }
    .title {
        color: #4A90E2;
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5em;
    }
    .subtitle {
        color: #333333;
        text-align: center;
        font-size: 1.1em;
        margin-bottom: 2em;
    }
    .result-box {
        background-color: #e6f7ff;
        padding: 1em;
        border-radius: 8px;
        margin-top: 1em;
        color: #333333;
    }
    </style>
    """, unsafe_allow_html=True)

# Main container for the app
with st.container():
    # Title and subtitle
    st.markdown("<div class='container'>", unsafe_allow_html=True)
    st.markdown("<div class='title'>Ingredient Classifier Bot</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Classify ingredients as 'Healthy', 'Moderate', or 'Unhealthy'</div>", unsafe_allow_html=True)

    # Load dataset and model
    try:
        df = pd.read_csv('ingredients - Sheet1 (4) (1).csv')
    except FileNotFoundError:
        st.error("The ingredients dataset file was not found. Please make sure the file is in the same directory as app.py.")
    
    try:
        model = joblib.load('ingredient_classifier_model.pkl')
    except FileNotFoundError:
        st.error("The classifier model file was not found. Please make sure the file is in the same directory as app.py.")

    # Define mapping for classification
    status_map = {2: "Healthy", 1: "Moderate", 0: "Unhealthy"}

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])

    # Track input fields and results in session state
    if "input_count" not in st.session_state:
        st.session_state.input_count = 1
    if "classification_results" not in st.session_state:
        st.session_state.classification_results = {}

    # Function to classify ingredients
    def classify_ingredients(ingredients):
        return [status_map[model.predict([ingredient])[0]] for ingredient in ingredients]

    # OCR function to read text from image
    def ocr_extract_text(image):
        image_np = np.array(image)
        results = reader.readtext(image_np, detail=0)
        ingredients = ",".join(results)
        return [ingredient.strip() for ingredient in ingredients.split(',')]

    # Option to upload an image
    uploaded_image = st.file_uploader("Upload an image of ingredients", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        ingredients_from_image = ocr_extract_text(image)
        st.write("Extracted Ingredients from Image:", ingredients_from_image)

        # Button to classify ingredients from image
        if st.button("Classify Ingredients from Image"):
            if ingredients_from_image:
                result = classify_ingredients(ingredients_from_image)
                st.session_state.classification_results["Image Ingredients"] = {
                    ingredient: classification for ingredient, classification in zip(ingredients_from_image, result)
                }

                # Display classification results
                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                st.markdown("**Classification Results for Image Ingredients:**", unsafe_allow_html=True)
                for ingredient, classification in st.session_state.classification_results["Image Ingredients"].items():
                    st.markdown(f"<p><b>{ingredient}</b>: {classification}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    # Loop to add text input fields for manual entry
    for i in range(st.session_state.input_count):
        user_input = st.text_input(f"Enter ingredients for Set {i+1} (comma-separated):", key=f"user_input_{i}")

        # Classify button for each set
        if st.button(f"Classify Ingredients for Set {i+1}"):
            if user_input:
                ingredients = [item.strip() for item in user_input.split(',')]
                result = classify_ingredients(ingredients)
                st.session_state.classification_results[f"Set {i+1}"] = {
                    ingredient: classification for ingredient, classification in zip(ingredients, result)
                }

                # Display classification results
                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                st.markdown(f"**Classification Results for Set {i+1}:**", unsafe_allow_html=True)
                for ingredient, classification in st.session_state.classification_results[f"Set {i+1}"].items():
                    st.markdown(f"<p><b>{ingredient}</b>: {classification}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    # Add a new input field for the next set
    if st.button("Add New Ingredient Set"):
        st.session_state.input_count += 1
        st.text_input(f"Enter ingredients for Set {st.session_state.input_count} (comma-separated):", key=f"user_input_{st.session_state.input_count-1}")

    st.markdown("</div>", unsafe_allow_html=True)
