import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Memuat dataset...")

# Load hasil preprocessing
df = pd.read_csv("data/processed/preprocessed.csv")

# Feature dan label
X = df["text"]
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train_vec, y_train)

# Evaluasi
pred = model.predict(X_test_vec)

acc = accuracy_score(y_test, pred)

print("Accuracy:", acc)

# Simpan model untuk tracking
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model tersimpan: model.pkl")