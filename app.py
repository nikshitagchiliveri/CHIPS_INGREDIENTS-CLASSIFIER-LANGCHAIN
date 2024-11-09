import streamlit as st
import pandas as pd
import joblib
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
    .add-button {
        display: flex;
        justify-content: center;
        margin-top: 2em;
    }
    </style>
    """, unsafe_allow_html=True)

# Main container for the app
with st.container():
    # Title and subtitle
    st.markdown("<div class='container'>", unsafe_allow_html=True)
    st.markdown("<div class='title'>Ingredient Classifier Bot</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Classify ingredients as 'Healthy', 'Moderate', or 'Unhealthy'</div>", unsafe_allow_html=True)

    # Load dataset
    df = pd.read_csv(r'C:\Users\NIKSHITA CHILIVERI\OneDrive\Desktop\INGREDIENTS-CLASSIFIER-LANGCHAIN\ingredients - Sheet1 (4) (1).csv')

    # Encode health status
    df['Healthy status'] = df['Healthy status'].map({'Healthy': 2, 'Moderate': 1, 'Unhealthy': 0})

    # Load the pre-trained model
    model = joblib.load(r'C:\Users\NIKSHITA CHILIVERI\OneDrive\Desktop\INGREDIENTS-CLASSIFIER-LANGCHAIN\ingredient_classifier_model.pkl')

    # Define the mapping for predictions
    status_map = {2: "Healthy", 1: "Moderate", 0: "Unhealthy"}

    # Initialize session state for tracking input fields and results
    if "input_count" not in st.session_state:
        st.session_state.input_count = 1
    if "classification_results" not in st.session_state:
        st.session_state.classification_results = {}

    def classify_ingredients(ingredients):
        # Predict and map the result to readable labels
        return [status_map[model.predict([ingredient])[0]] for ingredient in ingredients]

    # Input fields loop
    for i in range(st.session_state.input_count):
        # Input for each set
        user_input = st.text_input(f"Enter ingredients for Set {i+1} (comma-separated):", key=f"user_input_{i}")

        # Classify button for each set
        if st.button(f"Classify Ingredients for Set {i+1}"):
            if user_input:
                # Process ingredients
                ingredients = [item.strip() for item in user_input.split(',')]
                result = classify_ingredients(ingredients)

                # Store and display results under each input
                st.session_state.classification_results[f"Set {i+1}"] = {
                    ingredient: classification for ingredient, classification in zip(ingredients, result)
                }

                # Display results with styled box
                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                st.markdown(f"**Classification Results for Set {i+1}:**", unsafe_allow_html=True)
                for ingredient, classification in st.session_state.classification_results[f"Set {i+1}"].items():
                    st.markdown(f"<p><b>{ingredient}</b>: {classification}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    # Add a new input field for the next set
    with st.container():
        if st.button("Add New Ingredient Set"):
            st.session_state.input_count += 1
            st.text_input(f"Enter ingredients for Set {st.session_state.input_count} (comma-separated):", key=f"user_input_{st.session_state.input_count-1}")

    st.markdown("</div>", unsafe_allow_html=True)
