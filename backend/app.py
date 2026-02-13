from fastapi import FastAPI, UploadFile, File
import pickle
import joblib
import librosa
import numpy as np
import re
import string
import os
import cv2
import tensorflow as tf
import base64



app = FastAPI()

# ------------------------------
# LOAD MODELS
# ------------------------------

# Depression models
text_model = pickle.load(open("models/text_depression_model.pkl", "rb"))
vectorizer = pickle.load(open("models/text_depression_vectorizer.pkl", "rb"))
voice_model = joblib.load("models/voice_depression_model.pkl")

# Emotion models
text_emotion_model = pickle.load(open("models/text_emotion_model.pkl", "rb"))
text_emotion_vectorizer = pickle.load(open("models/text_emotion_vectorizer.pkl", "rb"))
voice_emotion_model = joblib.load("models/voice_emotion_model.pkl")
face_emotion_model = tf.keras.models.load_model("models/face_emotion_model.h5")

# ------------------------------
# Emotion and Face Label Mapping
# ------------------------------

emotion_labels = {
    0: "Anger",
    1: "Happiness",
    2: "Sadness",
    3: "Neutral",
    4: "Surprise",
    5: "Love"
}
face_labels = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise"
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

@app.post("/depression_text")
def predict_text_depression(text: str):

    text = clean_text(text)
    text_vec = vectorizer.transform([text])
    prediction = text_model.predict(text_vec)[0]

    return {
        "depression": int(prediction),
        "label": "Depressed" if prediction == 1 else "Not Depressed"
    }


@app.post("/depression_voice")
async def predict_voice_depression(file: UploadFile = File(...)):

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

@app.post("/depression_face")
async def depression_face(data: dict):

    image_data = data["image"]

    # Remove base64 header
    image_data = image_data.split(",")[1]

    img_bytes = base64.b64decode(image_data)

    with open("temp_depression.jpg", "wb") as f:
        f.write(img_bytes)

    img = cv2.imread("temp_depression.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(48,48))
    img = img / 255.0
    img = img.reshape(1,48,48,1)

    prediction = face_emotion_model.predict(img)
    emotion_id = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    os.remove("temp_depression.jpg")

    emotion_name = face_labels[emotion_id]

    # ---- Depression Risk Logic ----
    if emotion_name == "Sad" and confidence > 0.5:
        depression_status = "High Risk"
    elif emotion_name in ["Neutral", "Fear"] and confidence > 0.6:
        depression_status = "Moderate Risk"
    else:
        depression_status = "Low Risk"

    return {
        "emotion": emotion_name,
        "confidence": confidence,
        "depression_risk": depression_status
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

@app.post("/emotion_face")
async def emotion_face(file: UploadFile = File(...)):

    contents = await file.read()

    with open("temp.jpg", "wb") as f:
        f.write(contents)

    img = cv2.imread("temp.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(48,48))
    img = img / 255.0
    img = img.reshape(1,48,48,1)

    prediction = face_emotion_model.predict(img)
    emotion_id = int(tf.argmax(prediction, axis=1)[0])

    os.remove("temp.jpg")

    return {
        "emotion_id": emotion_id,
        "emotion": face_labels[emotion_id]
    }

@app.post("/emotion_webcam")
async def emotion_webcam(data: dict):

    image_data = data["image"]

    # Remove base64 header if present
    image_data = image_data.split(",")[1]

    img_bytes = base64.b64decode(image_data)

    with open("temp_webcam.jpg", "wb") as f:
        f.write(img_bytes)

    img = cv2.imread("temp_webcam.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48,48))
    img = img / 255.0
    img = img.reshape(1,48,48,1)

    prediction = face_emotion_model.predict(img)
    emotion_id = int(np.argmax(prediction))

    os.remove("temp_webcam.jpg")

    return {
        "emotion": face_labels[emotion_id]
    }




