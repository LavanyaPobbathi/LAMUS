#!/usr/bin/env python3
"""
LAMUS SaulLM-54B Complete Experiments
======================================
Test SaulLM-54B-Instruct with ALL THREE prompting strategies:
1. Zero-Shot (from Zero Shot Prompt.txt)
2. Few-Shot (from Few Shot Prompt.txt)
3. Chain-of-Thought (from CoT Shot Prompt.txt)

Uses 4-bit quantization to fit in GPU memory (~27GB instead of 108GB)

Run with: nohup python3 run_saulm54b_all_prompts.py > saulm54b_all_log.txt 2>&1 &
"""

import os
import sys
import json
import torch
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import warnings
import gc
warnings.filterwarnings('ignore')

os.environ['HF_HOME'] = '/home/lavanya/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/home/lavanya/.cache/huggingface'

# ============================================
HF_TOKEN = "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"
# ============================================

LABELS = ['Facts', 'Issue', 'Rule/Law/Holding', 'Analysis', 'Conclusion', 'Others']

# Prompt file paths (update if different)
PROMPT_FILES = {
    'Zero-Shot': 'Zero Shot Prompt.txt',
    'Few-Shot': 'Few Shot Prompt.txt', 
    'Chain-of-Thought': 'CoT Shot Prompt.txt'
}

def load_prompts():
    """Load all three prompts from text files"""
    prompts = {}
    
    print("\n📄 Loading prompts from files...")
    
    for prompt_name, filepath in PROMPT_FILES.items():
        # Try multiple possible paths
        possible_paths = [
            filepath,
            f"./{filepath}",
            f"./prompts/{filepath}",
            filepath.replace(' ', '_'),
            filepath.replace(' ', ''),
        ]
        
        found = False
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    prompts[prompt_name] = f.read().strip()
                print(f"  ✅ {prompt_name}: Loaded from {path} ({len(prompts[prompt_name])} chars)")
                found = True
                break
        
        if not found:
            print(f"  ❌ {prompt_name}: File not found - {filepath}")
            print(f"     Tried: {possible_paths}")
    
    return prompts

def extract_label(response):
    """Extract label from model response"""
    if not response:
        return "Others"
    
    response = response.strip()
    response_lower = response.lower()
    
    # Check for exact label at start
    label_map = {
        'facts': 'Facts',
        'fact': 'Facts',
        'issue': 'Issue',
        'rule/law/holding': 'Rule/Law/Holding',
        'rule': 'Rule/Law/Holding',
        'law': 'Rule/Law/Holding',
        'holding': 'Rule/Law/Holding',
        'analysis': 'Analysis',
        'conclusion': 'Conclusion',
        'others': 'Others',
        'other': 'Others'
    }
    
    # Check first word
    first_word = response_lower.split()[0] if response_lower.split() else ""
    first_word = first_word.strip('.,;:"\'-')
    
    if first_word in label_map:
        return label_map[first_word]
    
    # Check if any label appears in response
    for key, label in label_map.items():
        if key in response_lower:
            return label
    
    # Check for label after common patterns
    patterns = ['classification:', 'category:', 'answer:', 'label:']
    for pattern in patterns:
        if pattern in response_lower:
            after_pattern = response_lower.split(pattern)[-1].strip()
            first_word = after_pattern.split()[0] if after_pattern.split() else ""
            first_word = first_word.strip('.,;:"\'-')
            if first_word in label_map:
                return label_map[first_word]
    
    return "Others"

def load_saulm54b():
    """Load SaulLM-54B with 4-bit quantization"""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    
    print("\n📥 Loading SaulLM-54B-Instruct with 4-bit quantization...")
    print("   (This may take 10-15 minutes for first download)")
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        "Equall/SaulLM-54B-Instruct",
        token=HF_TOKEN,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        "Equall/SaulLM-54B-Instruct",
        token=HF_TOKEN,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ SaulLM-54B loaded successfully!")
    return model, tokenizer

def run_experiment(model, tokenizer, test_df, prompt_template, prompt_name):
    """Run experiment with a specific prompt"""
    
    print(f"\n{'='*60}")
    print(f"🔄 Running {prompt_name} Experiment")
    print(f"{'='*60}")
    
    predictions = []
    responses_log = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"{prompt_name}"):
        sentence = row['Sentence'][:500]  # Truncate long sentences
        
        # Format prompt with sentence
        if '{sentence}' in prompt_template.lower() or '{text}' in prompt_template.lower():
            # Replace placeholder
            prompt = prompt_template.replace('{sentence}', sentence)
            prompt = prompt.replace('{Sentence}', sentence)
            prompt = prompt.replace('{text}', sentence)
            prompt = prompt.replace('{TEXT}', sentence)
        else:
            # Append sentence at end
            prompt = prompt_template + f"\n\nSentence: {sentence}\n\nClassification:"
        
        try:
            inputs = tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode only new tokens
            response = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[-1]:],
                skip_special_tokens=True
            ).strip()
            
            pred = extract_label(response)
            
        except Exception as e:
            print(f"\n  ⚠️ Error at row {idx}: {str(e)[:50]}")
            response = ""
            pred = "Others"
        
        predictions.append(pred)
        
        # Log first 20 for debugging
        if idx < 20:
            responses_log.append({
                "idx": idx,
                "true": row['Label'],
                "pred": pred,
                "response": response[:100]
            })
    
    # Calculate accuracy
    true_labels = test_df['Label'].tolist()
    accuracy = accuracy_score(true_labels, predictions)
    
    return accuracy, predictions, responses_log

def main():
    print("="*70)
    print("LAMUS SaulLM-54B COMPLETE EXPERIMENTS")
    print(f"Started: {datetime.now()}")
    print("="*70)
    
    # Load prompts
    prompts = load_prompts()
    
    if not prompts:
        print("\n❌ No prompts loaded! Please check prompt files exist.")
        print("   Expected files:")
        for name, path in PROMPT_FILES.items():
            print(f"   - {path}")
        return
    
    # Load test data
    print("\n📊 Loading test data...")
    test_df = pd.read_csv('test_final.csv')
    print(f"  Test samples: {len(test_df)}")
    
    baseline = (test_df['Label'] == 'Facts').mean()
    print(f"  Baseline (majority class): {baseline*100:.2f}%")
    
    # Load model once
    model, tokenizer = load_saulm54b()
    
    # Run all experiments
    all_results = {}
    
    for prompt_name, prompt_template in prompts.items():
        accuracy, predictions, responses_log = run_experiment(
            model, tokenizer, test_df, prompt_template, prompt_name
        )
        
        all_results[prompt_name] = {
            'accuracy': accuracy,
            'predictions': predictions,
            'sample_responses': responses_log
        }
        
        print(f"\n📊 {prompt_name} Result: {accuracy*100:.2f}%")
        
        # Show classification report
        print(f"\n📋 Classification Report ({prompt_name}):")
        print(classification_report(
            test_df['Label'].tolist(), 
            predictions, 
            labels=LABELS, 
            zero_division=0
        ))
        
        # Clear cache between experiments
        torch.cuda.empty_cache()
        gc.collect()
    
    # Final Summary
    print("\n" + "="*70)
    print("📊 SAULM-54B FINAL RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n{'Prompt Strategy':<25} {'Accuracy':>15} {'vs Baseline':>15}")
    print("-"*55)
    
    for prompt_name, result in all_results.items():
        acc = result['accuracy']
        vs_baseline = acc - baseline
        print(f"{prompt_name:<25} {acc*100:>14.2f}% {vs_baseline*100:>+14.2f}%")
    
    print("-"*55)
    print(f"{'Baseline (Majority)':<25} {baseline*100:>14.2f}%")
    print(f"{'Best Prompting (Llama)':<25} {'75.89%':>15}")
    print(f"{'Fine-tuned (Llama)':<25} {'80.37%':>15}")
    
    # Find best SaulLM-54B result
    best_prompt = max(all_results.keys(), key=lambda x: all_results[x]['accuracy'])
    best_acc = all_results[best_prompt]['accuracy']
    
    print(f"\n🏆 Best SaulLM-54B Result: {best_prompt} with {best_acc*100:.2f}%")
    
    # Save all results
    results_to_save = {
        'model': 'SaulLM-54B-Instruct',
        'baseline': baseline,
        'results': {k: {'accuracy': v['accuracy'], 'sample_responses': v['sample_responses']} 
                   for k, v in all_results.items()},
        'best_prompt': best_prompt,
        'best_accuracy': best_acc,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('saulm54b_all_results.json', 'w') as f:
        json.dump(results_to_save, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: saulm54b_all_results.json")
    
    # Comparison with other models
    print("\n" + "="*70)
    print("📊 COMPARISON WITH OTHER MODELS")
    print("="*70)
    
    comparison = [
        ("Llama-3-8B (Zero-Shot)", 65.38),
        ("Llama-3-8B (Few-Shot)", 45.75),
        ("Llama-3-8B (CoT)", 75.89),
        ("Llama-3-8B (Fine-tuned)", 80.37),
    ]
    
    # Add SaulLM-54B results
    for prompt_name, result in all_results.items():
        comparison.append((f"SaulLM-54B ({prompt_name})", result['accuracy']*100))
    
    # Sort by accuracy
    comparison.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{'Model & Method':<35} {'Accuracy':>12}")
    print("-"*47)
    for model_name, acc in comparison:
        marker = "⭐" if "Fine-tuned" in model_name else ""
        print(f"{model_name:<35} {acc:>11.2f}% {marker}")
    
    print("\n" + "="*70)
    print(f"⏱️ Completed: {datetime.now()}")
    print("="*70)
    
    # Cleanup
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()