#!/usr/bin/env python3
"""
LAMUS Fine-Tuning Script with Legal Datasets
=============================================
Fine-tune Llama-3-8B using:
1. Your LAMUS training data (primary)
2. LegalBench tasks (supplementary)
3. Optional: Canadian Case Law

Goal: Reach 80-85% accuracy on legal text classification

Requirements:
pip install datasets peft trl bitsandbytes accelerate transformers

Run with: nohup python3 finetune_with_legalbench.py > finetune_log.txt 2>&1 &
"""

import os
import sys
import json
import torch
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

os.environ['HF_HOME'] = '/home/lavanya/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/home/lavanya/.cache/huggingface'

# ============================================
HF_TOKEN = "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"
# ============================================

LABELS = ['Facts', 'Issue', 'Rule/Law/Holding', 'Analysis', 'Conclusion', 'Others']

def load_lamus_data():
    """Load and prepare LAMUS training data"""
    print("\n📊 Loading LAMUS data...")
    
    train_df = pd.read_csv('train_final.csv')
    test_df = pd.read_csv('test_final.csv')
    
    # Create validation split
    train_df, val_df = train_test_split(
        train_df, test_size=0.1, random_state=42, stratify=train_df['Label']
    )
    
    print(f"  LAMUS Train: {len(train_df)}")
    print(f"  LAMUS Val: {len(val_df)}")
    print(f"  LAMUS Test: {len(test_df)}")
    
    return train_df, val_df, test_df

def load_legalbench_data():
    """Load relevant LegalBench tasks for supplementary training"""
    print("\n📊 Loading LegalBench data...")
    
    try:
        from datasets import load_dataset
        
        # LegalBench has many tasks - let's load relevant ones for classification
        # These tasks involve text classification similar to our task
        relevant_tasks = [
            'contract_nli_explicit_identification',
            'contract_nli_inclusion_of_verbally_conveyed_information', 
            'contract_nli_limited_use',
            'learned_hands_benefits',
            'learned_hands_courts',
            'learned_hands_crime',
            'learned_hands_family',
            'unfair_tos',
        ]
        
        all_data = []
        
        for task in relevant_tasks:
            try:
                ds = load_dataset("nguha/legalbench", task, trust_remote_code=True)
                
                if 'train' in ds:
                    for item in ds['train']:
                        # Convert to our format
                        if 'text' in item and 'label' in item:
                            all_data.append({
                                'text': str(item['text'])[:500],
                                'label': str(item['label']),
                                'source': f'legalbench_{task}'
                            })
                        elif 'premise' in item:
                            all_data.append({
                                'text': str(item.get('premise', ''))[:500],
                                'label': str(item.get('label', 'unknown')),
                                'source': f'legalbench_{task}'
                            })
                            
                print(f"  ✅ Loaded {task}")
                
            except Exception as e:
                print(f"  ⚠️ Could not load {task}: {str(e)[:50]}")
        
        print(f"  Total LegalBench samples: {len(all_data)}")
        return all_data
        
    except Exception as e:
        print(f"  ❌ Error loading LegalBench: {e}")
        return []

def prepare_training_data(train_df, legalbench_data=None, use_legalbench=False):
    """Prepare combined training data"""
    
    training_examples = []
    
    # Format LAMUS data
    print("\n🔄 Preparing training examples...")
    
    for _, row in train_df.iterrows():
        instruction = f"""Classify this legal sentence into exactly ONE category.

Categories:
- Facts: Events, evidence, what happened
- Issue: Legal questions to be resolved  
- Rule/Law/Holding: Legal rules, statutes, precedents
- Analysis: Legal reasoning, applying law to facts
- Conclusion: Final decisions, judgments
- Others: Procedural matters, other content

Sentence: {row['Sentence'][:500]}

Respond with ONLY the category name:"""
        
        response = row['Label']
        
        training_examples.append({
            'instruction': instruction,
            'response': response,
            'text': f"<s>[INST] {instruction} [/INST] {response}</s>"
        })
    
    print(f"  LAMUS examples: {len(training_examples)}")
    
    # Add LegalBench data if requested (helps with general legal understanding)
    if use_legalbench and legalbench_data:
        for item in legalbench_data[:500]:  # Limit to avoid imbalance
            instruction = f"""Analyze this legal text:

Text: {item['text'][:400]}

Classification:"""
            
            training_examples.append({
                'instruction': instruction,
                'response': item['label'],
                'text': f"<s>[INST] {instruction} [/INST] {item['label']}</s>"
            })
        
        print(f"  + LegalBench examples: {min(500, len(legalbench_data))}")
    
    print(f"  Total training examples: {len(training_examples)}")
    
    return training_examples

def finetune_llama():
    """Fine-tune Llama-3-8B with LoRA"""
    
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        BitsAndBytesConfig
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset
    
    print("="*70)
    print("LAMUS FINE-TUNING WITH LoRA")
    print(f"Started: {datetime.now()}")
    print("="*70)
    
    # Load data
    train_df, val_df, test_df = load_lamus_data()
    
    # Optionally load LegalBench
    USE_LEGALBENCH = False  # Set to True to include LegalBench data
    legalbench_data = None
    if USE_LEGALBENCH:
        legalbench_data = load_legalbench_data()
    
    # Prepare training data
    training_examples = prepare_training_data(train_df, legalbench_data, USE_LEGALBENCH)
    
    # Prepare validation data
    val_examples = []
    for _, row in val_df.iterrows():
        instruction = f"""Classify this legal sentence into exactly ONE category.

Categories:
- Facts: Events, evidence, what happened
- Issue: Legal questions to be resolved  
- Rule/Law/Holding: Legal rules, statutes, precedents
- Analysis: Legal reasoning, applying law to facts
- Conclusion: Final decisions, judgments
- Others: Procedural matters, other content

Sentence: {row['Sentence'][:500]}

Respond with ONLY the category name:"""
        
        val_examples.append({
            'instruction': instruction,
            'response': row['Label'],
            'text': f"<s>[INST] {instruction} [/INST] {row['Label']}</s>"
        })
    
    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_list(training_examples)
    val_dataset = Dataset.from_list(val_examples)
    
    # Load model with 4-bit quantization
    print("\n📥 Loading Llama-3-8B with 4-bit quantization...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        token=HF_TOKEN
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        token=HF_TOKEN,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    print("\n⚙️ Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Tokenize
    def tokenize(example):
        result = tokenizer(
            example['text'],
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        result['labels'] = result['input_ids'].copy()
        return result
    
    print("\n🔄 Tokenizing datasets...")
    train_tokenized = train_dataset.map(tokenize, remove_columns=train_dataset.column_names)
    val_tokenized = val_dataset.map(tokenize, remove_columns=val_dataset.column_names)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./lamus_finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=25,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=True,
        report_to="none",
        optim="paged_adamw_8bit",
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
    )
    
    # Train
    print("\n🚀 Starting fine-tuning...")
    print("   This may take 1-2 hours...")
    trainer.train()
    
    # Save
    print("\n💾 Saving fine-tuned model...")
    model.save_pretrained("./lamus_finetuned_final")
    tokenizer.save_pretrained("./lamus_finetuned_final")
    
    return model, tokenizer, test_df

def evaluate_model(model, tokenizer, test_df):
    """Evaluate fine-tuned model"""
    
    print("\n" + "="*70)
    print("📊 EVALUATING FINE-TUNED MODEL")
    print("="*70)
    
    predictions = []
    true_labels = test_df['Label'].tolist()
    
    model.eval()
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
        instruction = f"""Classify this legal sentence into exactly ONE category.

Categories:
- Facts: Events, evidence, what happened
- Issue: Legal questions to be resolved  
- Rule/Law/Holding: Legal rules, statutes, precedents
- Analysis: Legal reasoning, applying law to facts
- Conclusion: Final decisions, judgments
- Others: Procedural matters, other content

Sentence: {row['Sentence'][:500]}

Respond with ONLY the category name:"""
        
        input_text = f"<s>[INST] {instruction} [/INST]"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
        
        # Extract label
        pred = "Others"
        response_lower = response.lower()
        for label in LABELS:
            if label.lower() in response_lower:
                pred = label
                break
        
        predictions.append(pred)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    
    print(f"\n🎯 Fine-tuned Model Accuracy: {accuracy*100:.2f}%")
    print(f"📈 Previous best (prompting): 75.89%")
    print(f"📈 Improvement: {(accuracy - 0.7589)*100:+.2f}%")
    
    print("\n📋 Classification Report:")
    print(classification_report(true_labels, predictions, labels=LABELS, zero_division=0))
    
    # Per-class analysis
    print("\n📊 Per-Class Performance:")
    for label in LABELS:
        true_count = sum(1 for t in true_labels if t == label)
        pred_count = sum(1 for p in predictions if p == label)
        correct = sum(1 for t, p in zip(true_labels, predictions) if t == label and p == label)
        if true_count > 0:
            recall = correct / true_count * 100
            print(f"  {label}: {correct}/{true_count} ({recall:.1f}% recall)")
    
    # Save results
    results = {
        'accuracy': accuracy,
        'predictions': predictions,
        'true_labels': true_labels,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('finetune_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return accuracy

def main():
    print("="*70)
    print("LAMUS FINE-TUNING PIPELINE")
    print("="*70)
    
    # Check dependencies
    try:
        import peft
        import bitsandbytes
        from datasets import load_dataset
        print("✅ All dependencies installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nInstall with:")
        print("  pip install peft trl bitsandbytes accelerate datasets")
        return
    
    # Fine-tune
    model, tokenizer, test_df = finetune_llama()
    
    # Evaluate
    accuracy = evaluate_model(model, tokenizer, test_df)
    
    # Summary
    print("\n" + "="*70)
    print("FINE-TUNING COMPLETE")
    print("="*70)
    print(f"Final Accuracy: {accuracy*100:.2f}%")
    print(f"Previous Best: 75.89%")
    print(f"Target: 80-85%")
    
    if accuracy >= 0.80:
        print("\n🎉 TARGET ACHIEVED! Ready to label SCOTUS dataset.")
    elif accuracy >= 0.78:
        print("\n⚠️ Close to target. Consider:")
        print("   - More training epochs")
        print("   - Adding LegalBench data (set USE_LEGALBENCH=True)")
    else:
        print("\n⚠️ Below target. Consider:")
        print("   - Different hyperparameters")
        print("   - More training data from LegalBench")
        print("   - Trying SaulLM-54B")
    
    print("="*70)

if __name__ == "__main__":
    main()
