from fastapi import FastAPI, UploadFile, File
import pickle
import joblib
import librosa
import numpy as np
import re
import string
import os

app = FastAPI()

# ------------------------------
# LOAD MODELS
# ------------------------------

# Depression models
text_model = pickle.load(open("models/text_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
voice_model = joblib.load("models/voice_model.pkl")

# Emotion models
text_emotion_model = pickle.load(open("models/text_emotion_model.pkl", "rb"))
text_emotion_vectorizer = pickle.load(open("models/text_emotion_vectorizer.pkl", "rb"))
voice_emotion_model = joblib.load("models/voice_emotion_model.pkl")

# ------------------------------
# Emotion Label Mapping
# ------------------------------

emotion_labels = {
    0: "Anger",
    1: "Happiness",
    2: "Sadness",
    3: "Neutral",
    4: "Surprise",
    5: "Love"
}


# ------------------------------
# TEXT CLEANING
# ------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

# ------------------------------
# DEPRESSION ENDPOINTS
# ------------------------------

@app.post("/predict_text")
def predict_text(text: str):

    text = clean_text(text)
    text_vec = vectorizer.transform([text])
    prediction = text_model.predict(text_vec)[0]

    return {
        "depression": int(prediction),
        "label": "Depressed" if prediction == 1 else "Not Depressed"
    }


@app.post("/predict_voice")
async def predict_voice(file: UploadFile = File(...)):

    contents = await file.read()

    with open("temp.wav", "wb") as f:
        f.write(contents)

    audio, sample_rate = librosa.load("temp.wav")
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    features = np.mean(mfccs.T, axis=0).reshape(1, -1)

    prediction = voice_model.predict(features)[0]

    os.remove("temp.wav")

    return {
        "depression": int(prediction),
        "label": "Depressed" if prediction == 1 else "Not Depressed"
    }

# ------------------------------
# EMOTION ENDPOINTS
# ------------------------------

@app.post("/emotion_text")
def predict_text_emotion(text: str):

    text = clean_text(text)
    text_vec = text_emotion_vectorizer.transform([text])
    prediction = text_emotion_model.predict(text_vec)[0]

    return {
        "emotion_id": int(prediction),
        "emotion": emotion_labels.get(prediction, "Unknown")
    }


@app.post("/emotion_voice")
async def predict_voice_emotion(file: UploadFile = File(...)):

    contents = await file.read()

    with open("temp.wav", "wb") as f:
        f.write(contents)

    audio, sample_rate = librosa.load("temp.wav")
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    features = np.mean(mfccs.T, axis=0).reshape(1, -1)

    prediction = voice_emotion_model.predict(features)[0]

    os.remove("temp.wav")

    return {
        "emotion_id": int(prediction),
        "emotion": emotion_labels.get(prediction, "Unknown")
    }
