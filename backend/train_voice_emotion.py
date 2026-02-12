import os
import numpy as np
import librosa
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

os.makedirs("models", exist_ok=True)

DATASET_PATH = "dataset/voice/files"

# Emotion mapping based on filename
emotion_map = {
    "sad": 2,
    "euphoric": 1,
    "joyfully": 1,
    "surprised": 4
}

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

X = []
y = []

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".wav"):
            file_name = file.lower()

            # Detect emotion from filename
            for emotion in emotion_map:
                if emotion in file_name:
                    label = emotion_map[emotion]
                    file_path = os.path.join(root, file)

                    features = extract_features(file_path)
                    X.append(features)
                    y.append(label)

X = np.array(X)
y = np.array(y)

print("Total samples found:", len(X))

if len(X) == 0:
    print("No audio files detected! Check dataset path.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=300)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Voice Emotion Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(model, "models/voice_emotion_model.pkl")

print("Voice Emotion Model Saved!")
