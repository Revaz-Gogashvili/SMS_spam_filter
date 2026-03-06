import os
import pandas as pd
import numpy as np
import re
import torch
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding

# Force CPU for stability
os.environ["TRANSFORMERS_NO_TF"] = "1"

MODEL_NAME = "xlm-roberta-base" # Multilingual support
DATA_PATH = "../SMSSpamCollection"
MAX_LEN = 256

def basic_clean(text: str) -> str:
    t = re.sub(r'<.*?>', ' ', str(text))
    t = re.sub(r'https?://\S+|www\.\S+', '<URL>', t)
    t = re.sub(r'\b[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}\b', '<EMAIL>', t)
    return re.sub(r'\s+', ' ', t).strip().lower()

df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label_text", "text"], quoting=3)
df["label"] = df["label_text"].map({"ham": 0, "spam": 1})
df["text"] = df["text"].apply(basic_clean)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# Tokenization
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

train_ds = Dataset.from_pandas(train_df[["text", "label"]]).map(tokenize_fn, batched=True)
val_ds = Dataset.from_pandas(val_df[["text", "label"]]).map(tokenize_fn, batched=True)

# Model & Training
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    num_train_epochs=1, # Set to 1 for a quick test run in PyCharm
    use_cpu=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=DataCollatorWithPadding(tokenizer)
)

print("Fine-tuning Transformer (BERT/XLM-R)...")
trainer.train()

# Threshold Tuning
preds_logits = trainer.predict(val_ds).predictions
probs = softmax(preds_logits, axis=1)[:, 1] # Probability of being Spam
y_true = val_df["label"].values

# Finding the best threshold to maximize F1
precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-9)
best_threshold = thresholds[np.argmax(f1_scores)]

tuned_preds = (probs >= best_threshold).astype(int)

acc = accuracy_score(y_true, tuned_preds)
prec, rec, f1, _ = precision_recall_fscore_support(y_true, tuned_preds, average="binary")
print(f"\nTransformer Results (Tuned Threshold {best_threshold:.3f}):")
print(f" Accuracy: {acc:.4f}\n Precision: {prec:.4f}\n Recall: {rec:.4f}\n F1 Score: {f1:.4f}")

cm = confusion_matrix(y_true, tuned_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
disp.plot(cmap="Reds")
plt.title(f"Advanced: {MODEL_NAME}")
plt.savefig('xlmr_heatmap.png')
plt.show()
