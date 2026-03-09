#!/usr/bin/env python3
"""
LAMUS Experiment Runner - Final Version (nohup compatible)
==========================================================
Uses Serene's EXACT prompts from text files
Tests 4 models × 3 prompts = 12 experiments

Models:
- General Domain: Llama-3-8B, Qwen3-4B-Thinking
- Legal Domain: SaulLM-7B, law-LLM

Prompts (from files):
- Zero Shot Prompt.txt
- Few Shot Prompt.txt
- CoT Shot Prompt.txt

Run with: nohup python3 run_experiments_final.py > experiment_log.txt 2>&1 &
"""

import os
import sys
import json
import time
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
os.environ['HF_HOME'] = '/home/lavanya/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/home/lavanya/.cache/huggingface'

# ============================================
# UPDATE THIS WITH YOUR HUGGINGFACE TOKEN
# ============================================
HF_TOKEN = "HF_TOKEN_KEY"
# ============================================

# Labels (matching Serene's format)
LABELS = ['Facts', 'Issue', 'Rule/Law/Holding', 'Analysis', 'Conclusion', 'Others']

# Label mapping (Serene uses "Fact" and "Other", dataset uses "Facts" and "Others")
LABEL_MAP = {
    'Fact': 'Facts',
    'Facts': 'Facts',
    'Issue': 'Issue',
    'Rule/Law/Holding': 'Rule/Law/Holding',
    'Analysis': 'Analysis',
    'Conclusion': 'Conclusion',
    'Other': 'Others',
    'Others': 'Others'
}

# ============================================================================
# LOAD PROMPTS FROM FILES
# ============================================================================

def load_prompts():
    """Load Serene's exact prompts from text files"""
    
    prompt_files = {
        'Zero-Shot': 'Zero Shot Prompt.txt',
        'Few-Shot': 'Few Shot Prompt.txt',
        'Chain-of-Thought': 'CoT Shot Prompt.txt'
    }
    
    prompts = {}
    
    for prompt_name, filename in prompt_files.items():
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                prompts[prompt_name] = f.read().strip()
            print(f"  ✅ Loaded {prompt_name} from {filename}")
        except FileNotFoundError:
            print(f"  ❌ File not found: {filename}")
            sys.exit(1)
    
    return prompts

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_label(response, valid_labels=LABELS):
    """Extract classification label from model response based on Serene's format"""
    if not response:
        return "Others"
    
    response = str(response).strip()
    
    # Look for "Label: X" pattern (Serene's format)
    import re
    label_match = re.search(r'Label:\s*(\S+)', response, re.IGNORECASE)
    if label_match:
        found_label = label_match.group(1).strip()
        # Remove any trailing punctuation
        found_label = re.sub(r'[^\w/]', '', found_label)
        # Map to standard labels
        if found_label in LABEL_MAP:
            return LABEL_MAP[found_label]
        for label in LABELS:
            if label.lower() == found_label.lower():
                return label
    
    response_lower = response.lower()
    
    # Direct match
    for label in valid_labels:
        if label.lower() == response_lower:
            return label
        if response_lower.startswith(label.lower()):
            return label
    
    # Keyword matching for Serene's labels
    label_keywords = {
        'Facts': ['fact', 'facts', 'evidence', 'background', 'procedural history', 'testified', 'witness'],
        'Issue': ['issue', 'question', 'must determine', 'must resolve', 'whether'],
        'Rule/Law/Holding': ['rule', 'law', 'holding', 'statute', 'precedent', 'under state law', 'according to'],
        'Analysis': ['analysis', 'reasoning', 'application', 'because', 'applying', 'interpretation'],
        'Conclusion': ['conclusion', 'therefore', 'affirm', 'reverse', 'judgment', 'decision', 'we hold'],
        'Others': ['other', 'procedural', 'does not fit']
    }
    
    for label, keywords in label_keywords.items():
        for keyword in keywords:
            if keyword in response_lower:
                return label
    
    # Check if any label appears anywhere
    for label in valid_labels:
        if label.lower() in response_lower:
            return label
    
    return "Others"

def format_prompt_with_sentence(base_prompt, sentence):
    """Format prompt with the sentence to classify"""
    
    # Handle different placeholder formats
    if "[INSERT SENTENCE HERE]" in base_prompt:
        return base_prompt.replace("[INSERT SENTENCE HERE]", sentence)
    elif "{sentence}" in base_prompt:
        return base_prompt.format(sentence=sentence)
    else:
        # Append sentence at the end
        return f"{base_prompt}\n\nSentence: \"{sentence}\""

# ============================================================================
# MODEL CLASSES
# ============================================================================

class LlamaModel:
    """Llama-3-8B-Instruct - Uses chat template"""
    
    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("  Loading Llama-3-8B-Instruct...")
        self.name = "Llama-3-8B"
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            token=HF_TOKEN
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            token=HF_TOKEN,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("  ✅ Llama-3-8B loaded!")
    
    def generate(self, prompt, max_new_tokens=100):
        messages = [{"role": "user", "content": prompt}]
        
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[-1]:],
            skip_special_tokens=True
        )
        return response.strip()
    
    def cleanup(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()


class Qwen3ThinkingModel:
    """Qwen3-4B-Thinking-2507 - Uses chat template"""
    
    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("  Loading Qwen3-4B-Thinking-2507...")
        self.name = "Qwen3-Thinking"
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-4B-Thinking-2507",
            token=HF_TOKEN
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-4B-Thinking-2507",
            token=HF_TOKEN,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("  ✅ Qwen3-Thinking loaded!")
    
    def generate(self, prompt, max_new_tokens=100):
        messages = [{"role": "user", "content": prompt}]
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
        return response.strip()
    
    def cleanup(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()


class SaulLM7BModel:
    """SaulLM-7B-Instruct - Standard generation"""
    
    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("  Loading SaulLM-7B-Instruct...")
        self.name = "SaulLM-7B"
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Equall/Saul-7B-Instruct-v1",
            token=HF_TOKEN
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "Equall/Saul-7B-Instruct-v1",
            token=HF_TOKEN,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("  ✅ SaulLM-7B loaded!")
    
    def generate(self, prompt, max_new_tokens=100):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[-1]:],
            skip_special_tokens=True
        )
        return response.strip()
    
    def cleanup(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()


class LawLLMModel:
    """AdaptLLM/law-LLM - Standard generation"""
    
    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("  Loading law-LLM...")
        self.name = "law-LLM"
        self.tokenizer = AutoTokenizer.from_pretrained(
            "AdaptLLM/law-LLM",
            token=HF_TOKEN
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "AdaptLLM/law-LLM",
            token=HF_TOKEN,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("  ✅ law-LLM loaded!")
    
    def generate(self, prompt, max_new_tokens=100):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[-1]:],
            skip_special_tokens=True
        )
        return response.strip()
    
    def cleanup(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_experiment(model, model_name, base_prompt, prompt_name, test_df):
    """Run a single experiment with a model and prompt strategy"""
    
    print(f"\n{'='*60}")
    print(f"Running: {model_name} with {prompt_name}")
    print(f"Test samples: {len(test_df)}")
    print(f"{'='*60}")
    
    predictions = []
    responses_log = []
    true_labels = test_df['Label'].tolist()
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"{model_name}-{prompt_name}"):
        sentence = row['Sentence'][:500]  # Truncate very long sentences
        
        # Format prompt with sentence
        prompt = format_prompt_with_sentence(base_prompt, sentence)
        
        try:
            response = model.generate(prompt)
            pred = extract_label(response)
        except Exception as e:
            print(f"\n  Error at index {idx}: {e}")
            response = ""
            pred = "Others"
        
        predictions.append(pred)
        responses_log.append({
            "idx": idx,
            "sentence": sentence[:100],
            "true_label": row['Label'],
            "predicted": pred,
            "response": response[:200]
        })
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, labels=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(true_labels, predictions, labels=LABELS)
    
    result = {
        "model": model_name,
        "prompt": prompt_name,
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "predictions": predictions,
        "responses_sample": responses_log[:10],  # Save first 10 for debugging
        "timestamp": datetime.now().isoformat()
    }
    
    print(f"\n✅ {model_name} - {prompt_name}: {accuracy*100:.2f}% accuracy")
    
    # Show per-class performance
    print(f"   Per-class F1:")
    for label in LABELS:
        if label in report:
            f1 = report[label]['f1-score']
            print(f"     {label}: {f1*100:.1f}%")
    
    return result


def main():
    print("="*70)
    print("LAMUS EXPERIMENT RUNNER - FINAL VERSION")
    print(f"Started: {datetime.now()}")
    print("="*70)
    
    # Load prompts from files
    print("\n📄 Loading Serene's prompts from files...")
    prompts = load_prompts()
    
    # Load data
    print("\n📊 Loading data...")
    train_df = pd.read_csv('train_final.csv')
    test_df = pd.read_csv('test_final.csv')
    print(f"  Train: {len(train_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    # Calculate baseline
    baseline = (test_df['Label'] == 'Facts').mean()
    print(f"\n📈 Baseline (majority class 'Facts'): {baseline*100:.2f}%")
    
    # Models to test
    model_configs = [
        ("Llama-3-8B", LlamaModel),
        ("Qwen3-Thinking", Qwen3ThinkingModel),
        ("SaulLM-7B", SaulLM7BModel),
        ("law-LLM", LawLLMModel),
    ]
    
    print(f"\n🤖 Models to test: {len(model_configs)}")
    for name, _ in model_configs:
        print(f"   • {name}")
    
    print(f"\n📝 Prompts to test: {len(prompts)}")
    for name in prompts.keys():
        print(f"   • {name}")
    
    print(f"\n🧪 Total experiments: {len(model_configs) * len(prompts)}")
    
    # REMOVED input() for nohup compatibility
    print("\n🚀 Starting experiments automatically...")
    
    all_results = []
    
    for model_name, model_class in model_configs:
        print(f"\n{'#'*70}")
        print(f"# MODEL: {model_name}")
        print(f"{'#'*70}")
        
        try:
            # Load model
            model = model_class()
            
            # Run all prompt strategies
            for prompt_name, base_prompt in prompts.items():
                try:
                    result = run_experiment(
                        model=model,
                        model_name=model_name,
                        base_prompt=base_prompt,
                        prompt_name=prompt_name,
                        test_df=test_df
                    )
                    all_results.append(result)
                    
                    # Save intermediate results after each experiment
                    with open('experiment_results_intermediate.json', 'w') as f:
                        json.dump(all_results, f, indent=2, default=str)
                    print(f"  💾 Saved intermediate results")
                    
                except Exception as e:
                    print(f"\n❌ Error in {model_name} - {prompt_name}: {e}")
                    all_results.append({
                        "model": model_name,
                        "prompt": prompt_name,
                        "accuracy": None,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
            
            # Cleanup model to free GPU memory
            print(f"\n  🧹 Cleaning up {model_name}...")
            model.cleanup()
            
        except Exception as e:
            print(f"\n❌ Failed to load {model_name}: {e}")
            for prompt_name in prompts.keys():
                all_results.append({
                    "model": model_name,
                    "prompt": prompt_name,
                    "accuracy": None,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'experiment_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # ============================================
    # FINAL SUMMARY
    # ============================================
    print("\n" + "="*70)
    print("📊 EXPERIMENT SUMMARY")
    print("="*70)
    print(f"\nBaseline (majority class): {baseline*100:.2f}%")
    print(f"\n{'Model':<20} | {'Prompt':<18} | {'Accuracy':>10} | {'vs Baseline':>12}")
    print("-"*70)
    
    for result in all_results:
        if result.get('accuracy') is not None:
            acc = result['accuracy'] * 100
            diff = acc - (baseline * 100)
            status = "✅" if diff > 0 else "⚠️"
            print(f"{status} {result['model']:<18} | {result['prompt']:<18} | {acc:>9.2f}% | {diff:>+11.2f}%")
        else:
            print(f"❌ {result['model']:<18} | {result['prompt']:<18} | {'ERROR':>10} |")
    
    # Find best result
    valid_results = [r for r in all_results if r.get('accuracy') is not None]
    if valid_results:
        best = max(valid_results, key=lambda x: x['accuracy'])
        print(f"\n{'='*70}")
        print(f"🏆 BEST RESULT:")
        print(f"   Model: {best['model']}")
        print(f"   Prompt: {best['prompt']}")
        print(f"   Accuracy: {best['accuracy']*100:.2f}%")
        print(f"   Improvement over baseline: +{(best['accuracy'] - baseline)*100:.2f}%")
    
    print(f"\n💾 Results saved to: {results_file}")
    print(f"⏱️  Completed: {datetime.now()}")
    print("="*70)


if __name__ == "__main__":
    main()
