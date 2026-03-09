#!/usr/bin/env python3
"""
LAMUS Fine-Tuning Ablation Study v6
====================================
AGGRESSIVE MEMORY OPTIMIZATIONS:
1. Batch size = 1
2. Max length = 128
3. Gradient accumulation = 16
4. CUDA_VISIBLE_DEVICES=0 (single GPU only)
5. Disable DataParallel completely
6. Use 4-bit with double quant

Run with: CUDA_VISIBLE_DEVICES=0 python3 finetuning_ablation_v6.py
Or nohup: CUDA_VISIBLE_DEVICES=0 nohup python3 finetuning_ablation_v6.py > ablation_log.txt 2>&1 &
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
OUTPUT_DIR = "./ablation_results"

# AGGRESSIVE MEMORY SETTINGS
BATCH_SIZE = 1          # Minimum
GRAD_ACCUM = 16         # Compensate for small batch
MAX_LENGTH = 128        # Short sequences
# ============================================

LABELS = ['Facts', 'Issue', 'Rule/Law/Holding', 'Analysis', 'Conclusion', 'Others']

# Ablation configurations
ABLATION_CONFIGS = [
    # Learning rate ablation (fix epochs=3, lora_rank=16)
    {'name': 'lr_1e-5', 'lr': 1e-5, 'epochs': 3, 'lora_rank': 16},
    {'name': 'lr_5e-5', 'lr': 5e-5, 'epochs': 3, 'lora_rank': 16},
    {'name': 'lr_1e-4', 'lr': 1e-4, 'epochs': 3, 'lora_rank': 16},
    {'name': 'lr_2e-4', 'lr': 2e-4, 'epochs': 3, 'lora_rank': 16},
    
    # Epoch ablation (fix lr=2e-4, lora_rank=16)
    {'name': 'epoch_1', 'lr': 2e-4, 'epochs': 1, 'lora_rank': 16},
    {'name': 'epoch_2', 'lr': 2e-4, 'epochs': 2, 'lora_rank': 16},
    {'name': 'epoch_5', 'lr': 2e-4, 'epochs': 5, 'lora_rank': 16},
    
    # LoRA rank ablation (fix lr=2e-4, epochs=3)
    {'name': 'lora_8', 'lr': 2e-4, 'epochs': 3, 'lora_rank': 8},
    {'name': 'lora_32', 'lr': 2e-4, 'epochs': 3, 'lora_rank': 32},
]

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

def load_data():
    train_df = pd.read_csv('train_final.csv')
    test_df = pd.read_csv('test_final.csv')
    return train_df, test_df

def format_prompt(sentence):
    """Short prompt to save memory"""
    # Truncate long sentences
    if len(sentence) > 200:
        sentence = sentence[:200] + "..."
    return f"Classify: Facts/Issue/Rule/Analysis/Conclusion/Others\n\n{sentence}\n\nCategory:"

def run_single_ablation(config, train_df, test_df, tokenizer, results_file):
    """Run ablation with aggressive memory management"""
    
    config_name = config['name']
    
    print(f"\n{'='*60}")
    print(f"🔄 ABLATION: {config_name}")
    print(f"   LR: {config['lr']}, Epochs: {config['epochs']}, LoRA: {config['lora_rank']}")
    print(f"   GPU Memory: {get_gpu_memory()}")
    print(f"{'='*60}")
    
    # Check existing results
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            all_results = json.load(f)
        if config_name in all_results:
            print(f"   ⏭️ Already done ({all_results[config_name]['accuracy']*100:.2f}%)")
            return all_results[config_name]
    else:
        all_results = {}
    
    # Clear memory before loading
    clear_memory()
    print(f"   🧹 Memory cleared: {get_gpu_memory()}")
    
    # Import here to avoid memory issues
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from datasets import Dataset
    
    # 4-bit config with aggressive quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    print("   📥 Loading model...")
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
        r=config['lora_rank'],
        lora_alpha=config['lora_rank'] * 2,
        target_modules=["q_proj", "v_proj"],  # Reduced from 4 to 2 modules
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable params: {trainable:,}")
    print(f"   GPU after LoRA: {get_gpu_memory()}")
    
    # Prepare data
    print("   📝 Preparing data...")
    train_texts = []
    for _, row in train_df.iterrows():
        prompt = format_prompt(row['Sentence'])
        text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>{ASSISTANT_PREFIX}{row['Label']}<|eot_id|>"
        train_texts.append({'text': text})
    
    train_dataset = Dataset.from_list(train_texts)
    
    def tokenize_fn(examples):
        return tokenizer(examples['text'], truncation=True, max_length=MAX_LENGTH, padding='max_length')
    
    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=['text'])
    
    # Label masking
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
    
    # Training
    print("   🚀 Training...")
    training_args = TrainingArguments(
        output_dir=f"./ablation_models/{config_name}",
        num_train_epochs=config['epochs'],
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=config['lr'],
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        seed=42,
        fp16=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        max_grad_norm=1.0,
        # CRITICAL: Disable ddp/parallel
        ddp_find_unused_parameters=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    start_time = datetime.now()
    trainer.train()
    train_time = (datetime.now() - start_time).total_seconds()
    
    # Evaluate
    print("   📊 Evaluating...")
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="   Eval"):
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
    
    # Results
    true_labels = test_df['Label'].tolist()
    accuracy = accuracy_score(true_labels, predictions)
    
    result = {
        'config': config,
        'accuracy': accuracy,
        'training_time_seconds': train_time,
        'trainable_params': trainable,
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"\n   ✅ {config_name}: {accuracy*100:.2f}% ({train_time:.0f}s)")
    
    # Save
    all_results[config_name] = result
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Cleanup
    del model, trainer, train_dataset
    clear_memory()
    print(f"   🧹 Cleanup done: {get_gpu_memory()}")
    
    return result

def main():
    print("="*70)
    print("LAMUS FINE-TUNING ABLATION v6")
    print("(Aggressive Memory: batch=1, len=128, LoRA q+v only)")
    print(f"Started: {datetime.now()}")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("❌ No GPU!")
        return
    
    print(f"\n🖥️ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {get_gpu_memory()}")
    
    clear_memory()
    
    print("\n📊 Loading data...")
    train_df, test_df = load_data()
    print(f"   Train: {len(train_df)}, Test: {len(test_df)}")
    
    print("\n📥 Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_file = f"{OUTPUT_DIR}/ablation_results.json"
    
    print(f"\n🔄 Running {len(ABLATION_CONFIGS)} experiments...")
    
    for config in ABLATION_CONFIGS:
        try:
            run_single_ablation(config, train_df, test_df, tokenizer, results_file)
        except Exception as e:
            print(f"\n   ❌ Error in {config['name']}: {e}")
            import traceback
            traceback.print_exc()
            clear_memory()
    
    # Summary
    print("\n" + "="*70)
    print("📊 RESULTS")
    print("="*70)
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        if results:
            for name, r in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
                print(f"   {name:<15} {r['accuracy']*100:>7.2f}%")
            
            best = max(results.items(), key=lambda x: x[1]['accuracy'])
            print(f"\n🏆 Best: {best[0]} = {best[1]['accuracy']*100:.2f}%")
    
    print(f"\n⏱️ Done: {datetime.now()}")

if __name__ == "__main__":
    main()
