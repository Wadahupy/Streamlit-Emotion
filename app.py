import streamlit as st
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
import os

# Load models and labels
@st.cache_resource
def load_text_model():
    return joblib.load("text_emotion.pkl")

@st.cache_resource
def load_audio_model():
    return load_model('lstm_model.keras')

text_model = load_text_model()
audio_model = load_audio_model()

text_labels = text_model.classes_
audio_labels = ['angry', 'disgust', 'happy', 'fear', 'neutral', 'ps', 'sad']

# Preprocess audio for prediction
def preprocess_audio(file_path):
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        return np.expand_dims(np.expand_dims(mfccs, axis=-1), axis=0)
    except Exception as e:
        return str(e)

# Text emotion prediction
def predict_text_emotion(text):
    predicted_probabilities = text_model.predict_proba([text])[0]
    confidence = {emotion: float(prob) for emotion, prob in zip(text_labels, predicted_probabilities)}
    predicted_emotion = text_labels[np.argmax(predicted_probabilities)]
    return predicted_emotion, confidence

# Audio emotion prediction
def predict_audio_emotion(file_path):
    features = preprocess_audio(file_path)
    if isinstance(features, str):  # Error occurred
        return None, {"error": features}

    prediction = audio_model.predict(features)
    predicted_class = np.argmax(prediction)
    confidence = {label: float(prediction[0][i]) for i, label in enumerate(audio_labels)}
    predicted_emotion = audio_labels[predicted_class]

    return predicted_emotion, confidence

# Utility function to visualize confidence levels
def visualize_confidence(confidence, title):
    df = pd.DataFrame(list(confidence.items()), columns=["Emotion", "Confidence"])
    df = df.sort_values(by="Confidence", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df["Emotion"], df["Confidence"], color="skyblue")
    ax.set_title(title)
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Confidence Level")
    st.pyplot(fig)

# Utility function to clean up temporary files
def cleanup_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

# Main Streamlit app
st.title("Multimodal Emotion Detection")
st.subheader("Emotion Detection")
text_input = st.text_input("Enter text:")
audio_file = st.file_uploader("Upload an audio file:", type=["wav", "mp3"])


if text_input and audio_file:
    temp_file_path = f"temp_{audio_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(audio_file.read())

    text_emotion, text_confidence = predict_text_emotion(text_input)
    audio_emotion, audio_confidence = predict_audio_emotion(temp_file_path)

    if audio_confidence.get("error"):
        st.error(f"Error in audio processing: {audio_confidence['error']}")
    else:
        st.write(f"Predicted Text Emotion: **{text_emotion}**")
        st.write(f"Predicted Audio Emotion: **{audio_emotion}**")

        st.write("Text Confidence Scores:")
        visualize_confidence(text_confidence, "Text Emotion Confidence Levels")

        st.write("Audio Confidence Scores:")
        visualize_confidence(audio_confidence, "Audio Emotion Confidence Levels")

        # Combine final result
        combined_emotion = f"{text_emotion} (text) & {audio_emotion} (audio)"
        st.subheader(f"Final Combined Emotion: **{combined_emotion}**")

    cleanup_file(temp_file_path)
