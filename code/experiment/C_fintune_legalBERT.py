#!/usr/bin/env python3
"""
LAMUS LegalBERT Fine-tuning for Classification (Fixed)
=======================================================
Fixed version that handles PyTorch security issue

Run with: python3 finetune_legalbert_v2.py
"""

import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Fix for PyTorch security issue - set before importing transformers
os.environ["TRUST_REMOTE_CODE"] = "1"

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    BertTokenizer,
    BertForSequenceClassification
)
from datasets import Dataset

# ============================================
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
OUTPUT_DIR = "./legalbert_finetuned"
# ============================================

LABELS = ['Facts', 'Issue', 'Rule/Law/Holding', 'Analysis', 'Conclusion', 'Others']
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

def load_data():
    """Load train and test data"""
    print("📊 Loading data...")
    train_df = pd.read_csv('train_final.csv')
    test_df = pd.read_csv('test_final.csv')
    
    print(f"   Train samples: {len(train_df)}")
    print(f"   Test samples: {len(test_df)}")
    
    return train_df, test_df

def prepare_datasets(train_df, test_df, tokenizer):
    """Prepare HuggingFace datasets"""
    
    def tokenize_function(examples):
        return tokenizer(
            examples['Sentence'],
            padding='max_length',
            truncation=True,
            max_length=512
        )
    
    # Convert labels to IDs
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['label'] = train_df['Label'].map(LABEL2ID)
    test_df['label'] = test_df['Label'].map(LABEL2ID)
    
    # Create datasets
    train_dataset = Dataset.from_pandas(train_df[['Sentence', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['Sentence', 'label']])
    
    # Tokenize
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    return train_dataset, test_dataset

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    
    return {'accuracy': accuracy}

def main():
    print("="*70)
    print("LAMUS LEGALBERT FINE-TUNING (Fixed Version)")
    print(f"Started: {datetime.now()}")
    print("="*70)
    
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️ Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   PyTorch: {torch.__version__}")
    
    # Load data
    train_df, test_df = load_data()
    
    # Load tokenizer
    print(f"\n📥 Loading LegalBERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    # Load model with trust_remote_code and use safetensors
    print(f"📥 Loading LegalBERT model...")
    try:
        # Try loading with safetensors first
        model = BertForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            trust_remote_code=True,
            use_safetensors=True,  # Use safetensors format
            ignore_mismatched_sizes=True
        )
    except Exception as e1:
        print(f"   ⚠️ Safetensors failed, trying with torch_dtype...")
        try:
            model = BertForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels=len(LABELS),
                id2label=ID2LABEL,
                label2id=LABEL2ID,
                torch_dtype=torch.float32,
                ignore_mismatched_sizes=True
            )
        except Exception as e2:
            print(f"   ⚠️ Standard loading failed: {e2}")
            print("\n🔧 Trying alternative approach...")
            
            # Alternative: Load base BERT and use LegalBERT weights
            from transformers import BertModel, BertConfig
            
            # Load config
            config = BertConfig.from_pretrained(MODEL_NAME)
            config.num_labels = len(LABELS)
            config.id2label = ID2LABEL
            config.label2id = LABEL2ID
            
            # Initialize model with random classification head
            model = BertForSequenceClassification(config)
            
            # Load pretrained BERT weights (excluding classification head)
            pretrained = BertModel.from_pretrained(MODEL_NAME)
            model.bert = pretrained
            
            print("   ✅ Loaded LegalBERT with new classification head")
    
    model = model.to(device)
    print(f"   ✅ Model loaded!")
    
    # Prepare datasets
    print("\n📝 Preparing datasets...")
    train_dataset, test_dataset = prepare_datasets(train_df, test_df, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=50,
        report_to="none",
        seed=42,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train
    print("\n🚀 Starting training...")
    print(f"   Epochs: {training_args.num_train_epochs}")
    print(f"   Batch size: {training_args.per_device_train_batch_size}")
    print(f"   Learning rate: {training_args.learning_rate}")
    
    train_result = trainer.train()
    
    print(f"\n✅ Training complete!")
    print(f"   Training time: {train_result.metrics['train_runtime']:.2f}s")
    
    # Evaluate
    print("\n📊 Evaluating on test set...")
    eval_results = trainer.evaluate()
    print(f"   Test Accuracy: {eval_results['eval_accuracy']*100:.2f}%")
    
    # Get predictions for detailed analysis
    print("\n📋 Getting predictions...")
    predictions_output = trainer.predict(test_dataset)
    predictions = np.argmax(predictions_output.predictions, axis=-1)
    pred_labels = [ID2LABEL[p] for p in predictions]
    true_labels = test_df['Label'].tolist()
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(true_labels, pred_labels, labels=LABELS, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=LABELS)
    print("\n📋 Confusion Matrix:")
    cm_df = pd.DataFrame(cm, index=LABELS, columns=LABELS)
    print(cm_df)
    
    # Save model
    print(f"\n💾 Saving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save predictions
    results_df = test_df.copy()
    results_df['Predicted'] = pred_labels
    results_df['Correct'] = results_df['Label'] == results_df['Predicted']
    results_df.to_csv('legalbert_predictions.csv', index=False)
    
    # Save results summary
    accuracy = accuracy_score(true_labels, pred_labels)
    
    import json
    results = {
        'model': MODEL_NAME,
        'accuracy': accuracy,
        'classification_report': classification_report(true_labels, pred_labels, labels=LABELS, output_dict=True, zero_division=0),
        'confusion_matrix': cm.tolist(),
        'predictions': pred_labels,
        'training_args': {
            'epochs': training_args.num_train_epochs,
            'batch_size': training_args.per_device_train_batch_size,
            'learning_rate': training_args.learning_rate,
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open('legalbert_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: legalbert_results.json")
    print(f"💾 Predictions saved to: legalbert_predictions.csv")
    
    # Final summary
    print("\n" + "="*70)
    print("📊 LEGALBERT RESULTS SUMMARY")
    print("="*70)
    print(f"\n🏆 LegalBERT Accuracy: {accuracy*100:.2f}%")
    print(f"   Baseline (Majority): 61.98%")
    print(f"   Improvement: {(accuracy - 0.6198)*100:+.2f}%")
    
    # Compare with other models
    print("\n📊 Comparison with Other Models:")
    comparison = [
        ("Llama-3-8B (Fine-tuned)", 80.37),
        ("Llama-3-8B (CoT)", 75.89),
        ("SaulLM-54B (CoT)", 72.80),
        ("SaulLM-54B (Zero-Shot)", 67.39),
        ("Llama-3-8B (Zero-Shot)", 65.38),
        ("SaulLM-54B (Few-Shot)", 64.76),
        ("LegalBERT (Fine-tuned)", accuracy * 100),
    ]
    comparison.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, acc) in enumerate(comparison, 1):
        marker = "⭐ NEW" if "LegalBERT" in name else ""
        print(f"   {i}. {name}: {acc:.2f}% {marker}")
    
    print(f"\n⏱️ Completed: {datetime.now()}")

if __name__ == "__main__":
    main()