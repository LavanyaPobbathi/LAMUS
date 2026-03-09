#!/usr/bin/env python3
"""
LAMUS - Train ONLY the Best Model (85.16% Config)
==================================================
This is a simplified version of the ablation script that ONLY trains
the best configuration: lr=2e-4, epochs=3, lora_rank=16

This script worked before! Just running the winning config.

Run with: CUDA_VISIBLE_DEVICES=0 python3 D_train_best_only.py
Or nohup: CUDA_VISIBLE_DEVICES=0 nohup python3 D_train_best_only.py > train_best.log 2>&1 &
"""

import os
import sys
import gc

# CRITICAL: Set BEFORE importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
os.environ['HF_HOME'] = '/home/lavanya/.cache/huggingface'

import json
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================
HF_TOKEN = "HF_TOKEN_KEY"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
OUTPUT_DIR = "./best_model_85"  # Save best model here

# BEST CONFIG (85.16% accuracy)
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
LORA_RANK = 16

# Memory settings (same as working ablation)
BATCH_SIZE = 1
GRAD_ACCUM = 16
MAX_LENGTH = 128
# ============================================

LABELS = ['Facts', 'Issue', 'Rule/Law/Holding', 'Analysis', 'Conclusion', 'Others']
ASSISTANT_PREFIX = "<|start_header_id|>assistant<|end_header_id|>\n\n"

def clear_memory():
    """Aggressively clear GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

def get_gpu_memory():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"{allocated:.1f}/{total:.1f}GB"
    return "N/A"

def format_prompt(sentence):
    """Short prompt to save memory"""
    if len(sentence) > 200:
        sentence = sentence[:200] + "..."
    return f"Classify: Facts/Issue/Rule/Analysis/Conclusion/Others\n\n{sentence}\n\nCategory:"

def main():
    print("="*70)
    print("LAMUS - TRAIN BEST MODEL (85.16% Config)")
    print(f"Config: lr={LEARNING_RATE}, epochs={NUM_EPOCHS}, lora_rank={LORA_RANK}")
    print(f"Started: {datetime.now()}")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("❌ No GPU!")
        return
    
    print(f"\n🖥️ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {get_gpu_memory()}")
    
    clear_memory()
    
    # Load data
    print("\n📊 Loading data...")
    train_df = pd.read_csv('train_final.csv')
    test_df = pd.read_csv('test_final.csv')
    print(f"   Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Load tokenizer
    print("\n📥 Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Import training libraries
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from datasets import Dataset
    
    # 4-bit config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    print("\n📥 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    
    print(f"   GPU after model load: {get_gpu_memory()}")
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    
    # LoRA config
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable params: {trainable:,}")
    print(f"   GPU after LoRA: {get_gpu_memory()}")
    
    # Prepare training data
    print("\n📝 Preparing data...")
    train_texts = []
    for _, row in train_df.iterrows():
        prompt = format_prompt(row['Sentence'])
        text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>{ASSISTANT_PREFIX}{row['Label']}<|eot_id|>"
        train_texts.append({'text': text})
    
    train_dataset = Dataset.from_list(train_texts)
    
    def tokenize_fn(examples):
        return tokenizer(examples['text'], truncation=True, max_length=MAX_LENGTH, padding='max_length')
    
    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=['text'])
    
    # Label masking (KEY for 85.16%!)
    def add_labels(example):
        input_ids = example["input_ids"]
        labels = list(input_ids)
        text = tokenizer.decode(input_ids, skip_special_tokens=False)
        idx = text.find(ASSISTANT_PREFIX)
        if idx != -1:
            prefix = text[:idx + len(ASSISTANT_PREFIX)]
            prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
            for i in range(min(len(prefix_ids), len(labels))):
                labels[i] = -100
        example["labels"] = labels
        return example
    
    train_dataset = train_dataset.map(add_labels)
    
    # Training arguments
    print("\n🚀 Starting training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        seed=42,
        fp16=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        max_grad_norm=1.0,
        ddp_find_unused_parameters=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    start_time = datetime.now()
    trainer.train()
    train_time = datetime.now() - start_time
    
    print(f"\n✅ Training complete! Time: {train_time}")
    
    # Save model
    print(f"\n💾 Saving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Evaluate on full test set
    print("\n📊 Evaluating on test set...")
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
            prompt = format_prompt(row['Sentence'])
            input_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>{ASSISTANT_PREFIX}"
            
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
            
            pred = "Others"
            for label in LABELS:
                if label.lower() in response.lower():
                    pred = label
                    break
            predictions.append(pred)
    
    # Calculate accuracy
    true_labels = test_df['Label'].tolist()
    accuracy = accuracy_score(true_labels, predictions)
    
    # Save summary
    summary = {
        "model": "Llama-3-8B Fine-tuned (Best Config)",
        "config": {
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "lora_rank": LORA_RANK,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "label_masking": True
        },
        "accuracy": accuracy,
        "training_time": str(train_time),
        "trainable_params": trainable,
        "timestamp": datetime.now().isoformat(),
        "output_dir": OUTPUT_DIR
    }
    
    with open(f"{OUTPUT_DIR}/training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Final summary
    print("\n" + "="*70)
    print("🏆 TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📁 Model saved to: {OUTPUT_DIR}")
    print(f"⏱️ Training time: {train_time}")
    print(f"📊 Test Accuracy: {accuracy*100:.2f}%")
    print(f"   Expected: ~85.16%")
    print(f"   Baseline: 61.98%")
    print(f"   Improvement: +{(accuracy - 0.6198)*100:.2f}%")
    
    print(f"\n🚀 Next step: Run labeling script")
    print(f"   nohup python3 D_label_roberts_court.py > roberts_labeling_log.txt 2>&1 &")
    
    # Cleanup
    del model, trainer
    clear_memory()
    print(f"\n✅ Done! {datetime.now()}")

if __name__ == "__main__":
    main()
