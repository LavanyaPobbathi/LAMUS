#!/usr/bin/env python3
"""
LAMUS - Complete Ablation Grid for WikiSQL-Style Figure
========================================================
Runs ALL 36 combinations (skips 8 already done):
- Learning Rates: 1e-5, 5e-5, 1e-4, 2e-4
- LoRA Ranks: 8, 16, 32  
- Epochs: 1, 3, 5

Run: CUDA_VISIBLE_DEVICES=0 nohup python3 F_run_ablation_grid.py > ablation_grid.log 2>&1 &
Monitor: tail -f ablation_grid.log

Time estimate: ~12-14 hours (28 new experiments × ~25 min each)
"""

import os
import sys
import gc
import json
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:64')
os.environ.setdefault('HF_HOME', '/home/lavanya/.cache/huggingface')

# ============================================
# CONFIGURATION
# ============================================
HF_TOKEN = "HF_TOKEN_KEY"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
OUTPUT_DIR = "./ablation_results"

# Complete grid for WikiSQL-style figure
LEARNING_RATES = [1e-5, 5e-5, 1e-4, 2e-4]
LORA_RANKS = [8, 16, 32]
EPOCHS_LIST = [1, 3, 5]

# Fixed parameters
BATCH_SIZE = 1
GRAD_ACCUM = 16
MAX_LENGTH = 128
# ============================================

LABELS = ['Facts', 'Issue', 'Rule/Law/Holding', 'Analysis', 'Conclusion', 'Others']
ASSISTANT_PREFIX = "<|start_header_id|>assistant<|end_header_id|>\n\n"

# Already completed experiments (mapped to epochs 1,3,5 grid)
COMPLETED_RESULTS = {
    # (learning_rate, lora_rank, epochs): accuracy
    (2e-4, 16, 3): 85.16,
    (2e-4, 8, 3): 84.70,
    (1e-4, 16, 3): 84.54,
    (2e-4, 32, 3): 84.39,
    (2e-4, 16, 5): 83.93,
    (5e-5, 16, 3): 83.15,
    (2e-4, 16, 1): 82.69,
    (1e-5, 16, 3): 75.12,
}


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"{allocated:.1f}/{total:.1f}GB"
    return "N/A"


def format_prompt(sentence):
    if len(sentence) > 200:
        sentence = sentence[:200] + "..."
    return f"Classify: Facts/Issue/Rule/Analysis/Conclusion/Others\n\n{sentence}\n\nCategory:"


def run_experiment(lr, rank, epochs, train_df, test_df, tokenizer):
    """Run a single ablation experiment"""
    
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from datasets import Dataset
    
    config_name = f"lr{lr}_r{rank}_e{epochs}"
    print(f"\n{'='*60}")
    print(f"🔬 Running: LR={lr}, Rank={rank}, Epochs={epochs}")
    print(f"{'='*60}")
    
    clear_memory()
    
    # 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    print("📥 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    
    # LoRA config
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable params: {trainable:,}")
    
    # Prepare training data with label masking
    train_texts = []
    for _, row in train_df.iterrows():
        prompt = format_prompt(row['Sentence'])
        text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>{ASSISTANT_PREFIX}{row['Label']}<|eot_id|>"
        train_texts.append({'text': text})
    
    train_dataset = Dataset.from_list(train_texts)
    
    def tokenize_fn(examples):
        return tokenizer(examples['text'], truncation=True, max_length=MAX_LENGTH, padding='max_length')
    
    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=['text'])
    
    # Label masking (critical!)
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
    temp_output = f"./temp_ablation_{config_name}"
    training_args = TrainingArguments(
        output_dir=temp_output,
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=lr,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        seed=42,
        fp16=True,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        optim="adamw_torch",
        max_grad_norm=1.0,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    print("🚀 Training...")
    start_time = datetime.now()
    trainer.train()
    train_time = datetime.now() - start_time
    print(f"   Training time: {train_time}")
    
    # Evaluate
    print("📊 Evaluating...")
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Eval", leave=False):
            prompt = format_prompt(row['Sentence'])
            input_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>{ASSISTANT_PREFIX}"
            
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
            
            pred = "Others"
            for label in LABELS:
                if label.lower() in response.lower():
                    pred = label
                    break
            predictions.append(pred)
    
    accuracy = accuracy_score(test_df['Label'].tolist(), predictions)
    
    # Cleanup
    del model, trainer
    clear_memory()
    
    import shutil
    if os.path.exists(temp_output):
        shutil.rmtree(temp_output)
    
    print(f"   ✅ Accuracy: {accuracy*100:.2f}%")
    
    return {
        'learning_rate': lr,
        'lora_rank': rank,
        'epochs': epochs,
        'accuracy': round(accuracy * 100, 2),
        'trainable_params': trainable,
        'train_time': str(train_time),
    }


def main():
    print("="*70)
    print("LAMUS - COMPLETE ABLATION GRID")
    print(f"Started: {datetime.now()}")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("❌ No GPU!")
        return
    
    print(f"\n🖥️ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {get_gpu_memory()}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
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
    
    # Generate all combinations
    all_configs = []
    for lr in LEARNING_RATES:
        for rank in LORA_RANKS:
            for epochs in EPOCHS_LIST:
                all_configs.append((lr, rank, epochs))
    
    print(f"\n📋 Total configurations: {len(all_configs)}")
    print(f"   Grid: {len(LEARNING_RATES)} LRs × {len(LORA_RANKS)} Ranks × {len(EPOCHS_LIST)} Epochs")
    
    # Load/initialize results
    results_file = os.path.join(OUTPUT_DIR, "ablation_grid_results.json")
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            all_results = json.load(f)
        print(f"\n📂 Loaded {len(all_results)} existing results")
    else:
        all_results = []
    
    # Track done configs
    done_configs = set()
    for r in all_results:
        done_configs.add((r['learning_rate'], r['lora_rank'], r['epochs']))
    
    # Add previous results
    for (lr, rank, ep), acc in COMPLETED_RESULTS.items():
        if (lr, rank, ep) not in done_configs:
            all_results.append({
                'learning_rate': lr,
                'lora_rank': rank,
                'epochs': ep,
                'accuracy': acc,
                'trainable_params': None,
                'train_time': 'previous',
            })
            done_configs.add((lr, rank, ep))
            print(f"   ✅ Added previous: LR={lr}, Rank={rank}, Epochs={ep} -> {acc}%")
    
    # Save with previous
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Find remaining
    remaining = [c for c in all_configs if c not in done_configs]
    print(f"\n🔬 Remaining experiments: {len(remaining)}/{len(all_configs)}")
    
    if remaining:
        est_hours = len(remaining) * 25 / 60
        print(f"   Estimated time: ~{est_hours:.1f} hours")
    else:
        print("   ✅ All experiments complete!")
    
    # Run remaining
    for i, (lr, rank, epochs) in enumerate(remaining):
        print(f"\n{'='*70}")
        print(f"📊 Progress: {i+1}/{len(remaining)} | Total: {len(done_configs)+i+1}/{len(all_configs)}")
        print(f"{'='*70}")
        
        try:
            result = run_experiment(lr, rank, epochs, train_df, test_df, tokenizer)
            all_results.append(result)
            done_configs.add((lr, rank, epochs))
            
            # Save after each
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"   💾 Saved ({len(all_results)} total)")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    print("\n" + "="*70)
    print("📊 COMPLETE ABLATION GRID RESULTS")
    print("="*70)
    
    df = pd.DataFrame(all_results)
    
    for epoch in EPOCHS_LIST:
        print(f"\n📈 Epoch = {epoch}:")
        print("-" * 55)
        print(f"   {'LR':<12} {'Rank 8':>10} {'Rank 16':>10} {'Rank 32':>10}")
        print("   " + "-"*45)
        
        epoch_data = df[df['epochs'] == epoch]
        for lr in LEARNING_RATES:
            row = f"   {lr:<12}"
            for rank in [8, 16, 32]:
                val = epoch_data[(epoch_data['learning_rate'] == lr) & 
                                (epoch_data['lora_rank'] == rank)]['accuracy']
                if len(val) > 0:
                    row += f"{val.values[0]:>10.2f}"
                else:
                    row += f"{'---':>10}"
            print(row)
    
    # Save CSV
    csv_file = os.path.join(OUTPUT_DIR, "ablation_grid_results.csv")
    df.to_csv(csv_file, index=False)
    print(f"\n💾 Saved: {csv_file}")
    
    # Best
    best_idx = df['accuracy'].idxmax()
    best = df.loc[best_idx]
    print(f"\n🏆 Best: LR={best['learning_rate']}, Rank={best['lora_rank']}, Epochs={best['epochs']} -> {best['accuracy']:.2f}%")
    
    print(f"\n✅ Complete! {datetime.now()}")
    print("   Next: python3 F_create_final_ablation_figure.py")


if __name__ == "__main__":
    main()
