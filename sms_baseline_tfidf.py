import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay

DATA_PATH = "SMSSpamCollection"

def basic_clean(text: str) -> str:
    t = re.sub(r'<.*?>', ' ', str(text))  # Remove HTML
    t = re.sub(r'https?://\S+|www\.\S+', '<URL>', t)
    t = re.sub(r'\b[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}\b', '<EMAIL>', t)
    return re.sub(r'\s+', ' ', t).strip().lower()

df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label_text", "text"], quoting=3)
df["label"] = df["label_text"].map({"ham": 0, "spam": 1})
df["text"] = df["text"].apply(basic_clean)
df = df.dropna()

X_train, X_val, y_train, y_val = train_test_split(
    df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
)

# ngram_range(1,2) means it looks at single words and two-word pairs
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2)),
    ("clf", LogisticRegression(class_weight='balanced'))
])

print("Training TF-IDF Baseline...")
pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_val)
acc = accuracy_score(y_val, preds)
prec, rec, f1, _ = precision_recall_fscore_support(y_val, preds, average="binary")

print(f"\nResults:\n Accuracy: {acc:.4f}\n Precision: {prec:.4f}\n Recall: {rec:.4f}\n F1 Score: {f1:.4f}")

# 5. Visualization
cm = confusion_matrix(y_val, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
disp.plot(cmap="Blues")
plt.title("Baseline: TF-IDF + Logistic Regression")
plt.savefig('baseline_heatmap.png')
plt.show()