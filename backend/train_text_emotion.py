import pandas as pd
import re
import string
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("dataset/text/tweet_emotions.csv")

# Keep only required columns
df = df[["sentiment", "content"]]
df.columns = ["label", "text"]

# Keep only major emotions
allowed_emotions = ["anger", "happiness", "sadness", "surprise", "neutral", "love"]

df = df[df["label"].isin(allowed_emotions)]

# Encode labels
label_mapping = {
    "anger": 0,
    "happiness": 1,
    "sadness": 2,
    "neutral": 3,
    "surprise": 4,
    "love": 5
}

df["label"] = df["label"].map(label_mapping)

#oversampling block
from sklearn.utils import resample

# Separate majority and minority class
df_major = df[df.label != 0]   # All except anger
df_anger = df[df.label == 0]   # Only anger

print("Before oversampling:")
print(df["label"].value_counts())

# Oversample anger to 2000 samples
df_anger_upsampled = resample(
    df_anger,
    replace=True,
    n_samples=2000,
    random_state=42
)

# Combine back
df = pd.concat([df_major, df_anger_upsampled])

print("After oversampling:")
print(df["label"].value_counts())


# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

df["text"] = df["text"].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1,2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LinearSVC(class_weight="balanced")

model.fit(X_train_vec, y_train)


y_pred = model.predict(X_test_vec)


print("Text Emotion Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

pickle.dump(model, open("models/text_emotion_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/text_emotion_vectorizer.pkl", "wb"))

print("Text Emotion Model Saved!")
