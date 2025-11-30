from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


# Detect device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load and encode training data
train_df = pd.read_csv("train.csv")
label_encoder = LabelEncoder()
train_df["label_encoded"] = label_encoder.fit_transform(train_df["label"])

# 2. Prepare Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df[["text", "label_encoded"]])
dataset = DatasetDict({"train": train_dataset})

model_dir = "legalbert_fewshot_model"

# 3. Load LegalBERT and tokenizer
if os.path.exists(model_dir):
    print("Loading existing model from disk...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
else:
    print("Training new model...")
    model_name = "nlpaueb/legal-bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))
    model.to(device)  # Move model to device

    # 4. Tokenization function
    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

    dataset = dataset.map(tokenize)
    dataset = dataset.rename_column("label_encoded", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,
        per_device_train_batch_size=4,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="no",
        load_best_model_at_end=False,
    )

    # 6. Trainer and Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
    )
    trainer.train()
    # Save model and tokenizer
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

# 7. Load test data
test_df = pd.read_csv("shuffled.csv")
if "text" not in test_df.columns:
    test_df.columns = ["text"]

# Remove empty or NaN rows
test_df = test_df.dropna(subset=["text"])
test_df = test_df[test_df["text"].str.strip().astype(bool)]

test_texts = list(test_df["text"])
true_labels = test_df["label"] if "label" in test_df.columns else None

# 8. Tokenize test data
test_encodings = tokenizer(test_texts, truncation=True, padding=True, return_tensors="pt")
input_ids = test_encodings["input_ids"]
attention_mask = test_encodings["attention_mask"]
batch_size = 8  # Adjust as needed

from torch.utils.data import DataLoader, TensorDataset

test_dataset = TensorDataset(input_ids, attention_mask)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

model.eval()
preds = []

with torch.no_grad():
    for batch in test_loader:
        input_ids_batch, attention_mask_batch = batch
        input_ids_batch = input_ids_batch.to(device)
        attention_mask_batch = attention_mask_batch.to(device)

        outputs = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)
        batch_preds = torch.argmax(outputs.logits, dim=1)
        preds.extend(batch_preds.cpu().numpy())

if len(preds) != len(test_df):
    print(f"Warning: Number of predictions ({len(preds)}) does not match number of test rows ({len(test_df)})")

# 10. Save predictions
test_df["predicted_label"] = label_encoder.inverse_transform(preds)
test_df.to_csv("fewshot_predictions.csv", index=False)
print("Predictions saved to few_shot_predictions.csv")