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
df = pd.read_excel("dataset/text/Depression_Text.xlsx")

# Select required columns
df = df[["text", "label"]]

# Remove missing labels
df = df.dropna(subset=["label"])

# Convert label to integer
df["label"] = df["label"].astype(int)

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

df["text"] = df["text"].apply(clean_text)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_vec, y_train)

# Predict
y_pred = model.predict(X_test_vec)

print("Text Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save
pickle.dump(model, open("models/text_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

print("✅ Text model saved successfully!")
