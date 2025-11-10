"""
Sentiment Analysis Project
Author: Rahil Khan
Description:
A full sentiment analysis pipeline using NLTK and scikit-learn.
It classifies text (tweets, product reviews, etc.) as Positive, Negative, or Neutral
based on training data. The model uses TF-IDF vectorization and Logistic Regression.
"""

# ==============================
# 1. Import Required Libraries
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import string
import re

# Download NLTK resources (run once)
nltk.download('stopwords')
nltk.download('wordnet')

# ==============================
# 2. Load Dataset
# ==============================
# Path to your local dataset
DATA_PATH = r"C:\Users\rahil\OneDrive\Documents\testdata.manual.2009.06.14.csv"

# Load CSV (Sentiment140 format)
df = pd.read_csv(DATA_PATH, encoding="latin-1", header=None)
df.columns = ["sentiment", "id", "date", "query", "user", "text"]

# Keep only the sentiment and text columns
df = df[["sentiment", "text"]]

# Sentiment140 labels: 0 = Negative, 2 = Neutral, 4 = Positive
df["sentiment"] = df["sentiment"].replace({0: "negative", 2: "neutral", 4: "positive"})

print("✅ Dataset loaded successfully!")
print(df.head())

# ==============================
# 3. Data Cleaning
# ==============================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove links
    text = re.sub(r"\@\w+|\#", "", text)  # remove mentions and hashtags
    text = re.sub(r"[^A-Za-z\s]", "", text)  # keep only letters
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and w not in string.punctuation]
    return " ".join(words)

print("🧹 Cleaning text data...")
df["clean_text"] = df["text"].apply(clean_text)

# ==============================
# 4. Exploratory Data Analysis
# ==============================
print("\n📊 Class distribution:")
print(df["sentiment"].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="sentiment", hue="sentiment", palette="coolwarm", legend=False)
plt.title("Sentiment Distribution")
plt.show()

# ==============================
# 5. Feature Extraction (TF-IDF)
# ==============================
print("\n🔠 Converting text to TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["clean_text"])
y = df["sentiment"]

# ==============================
# 6. Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 7. Model Training
# ==============================
print("\n🤖 Training Logistic Regression model...")
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# ==============================
# 8. Model Evaluation
# ==============================
print("\n📈 Evaluating model...")
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\n✅ Accuracy Score:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

# ==============================
# 9. Test with Custom Inputs
# ==============================
def predict_sentiment(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    return pred

print("\n💬 Try sample predictions:")
samples = [
    "I love this product! It's amazing and works perfectly.",
    "This is the worst experience I've ever had.",
    "The movie was okay, not great but not terrible either."
]

for s in samples:
    print(f"Text: {s}\nPredicted Sentiment: {predict_sentiment(s)}\n")

# ==============================
# 10. Save Model & Vectorizer (Optional)
# ==============================
# Uncomment to save trained model and vectorizer
# import joblib
# joblib.dump(model, "sentiment_model.pkl")
# joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\n🎉 Sentiment Analysis complete!")
