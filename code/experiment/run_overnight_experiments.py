# Save as run_professor_models_final.py
import os
os.environ['HF_HOME'] = '/home/lavanya/.cache/huggingface'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import json
from datetime import datetime
import traceback
import gc
import time

HF_TOKEN = 'hf_oxD'

# ONLY models specified by professor (excluding those already tested)
PROFESSOR_MODELS = [
    ("Llama-3-8B", "meta-llama/Meta-Llama-3-8B-Instruct"),  # General domain
    ("SaulLM-7B", "Equall/Saul-7B-Instruct-v1"),  # Legal domain
    ("Law-LLM", "AdaptLLM/law-LLM"),  # Legal domain
    # Note: Gemini already tested via API (6.5% accuracy)
    # Note: SaulLM-54B skipped (too large - would need 108GB)
    # Note: Qwen3-Thinking-2507 may not be available
]

# Test mode
TEST_MODE = False  # Change to False for full run
test_df = pd.read_csv('test_final.csv')
train_df = pd.read_csv('train_final.csv')

if TEST_MODE:
    test_df = test_df.head(5)
    print(f"TEST MODE: {len(test_df)} samples")
else:
    print(f"FULL RUN: {len(test_df)} samples")

def get_few_shot_examples(n=2):
    """Get examples for few-shot"""
    examples = []
    for label in ['Facts', 'Issue', 'Rule/Law/Holding', 'Analysis', 'Conclusion', 'Others']:
        samples = train_df[train_df['Label'] == label].head(n)
        for _, row in samples.iterrows():
            examples.append(f"Sentence: {row['Sentence'][:150]}\nCategory: {label}")
    return "\n\n".join(examples[:n*3])

def extract_label(response):
    """Extract label from response"""
    response_clean = response.strip().lower()
    if "facts" in response_clean:
        return "Facts"
    elif "issue" in response_clean:
        return "Issue"
    elif any(word in response_clean for word in ["rule", "law", "holding"]):
        return "Rule/Law/Holding"
    elif "analysis" in response_clean:
        return "Analysis"
    elif "conclusion" in response_clean:
        return "Conclusion"
    return "Others"

def run_all_experiments(model_name, model_path):
    """Run all 3 experiment types as per professor"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('='*60)
    
    results = {}
    
    try:
        # Load model
        print("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            token=HF_TOKEN,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 1. ZERO-SHOT
        print("\n1. Running Zero-shot...")
        predictions = []
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Zero-shot"):
            prompt = f"""Classify this legal text: Facts, Issue, Rule/Law/Holding, Analysis, Conclusion, or Others.

Text: {row['Sentence'][:400]}

Category:"""
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=10, temperature=0, do_sample=False)
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
            predictions.append(extract_label(response))
        
        accuracy = accuracy_score(test_df['Label'], predictions)
        results['zero_shot'] = accuracy
        print(f"Zero-shot Accuracy: {accuracy:.3f}")
        
        # 2. FEW-SHOT
        print("\n2. Running Few-shot...")
        few_shot_examples = get_few_shot_examples(2)
        predictions = []
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Few-shot"):
            prompt = f"""Examples:
{few_shot_examples}

Text: {row['Sentence'][:300]}
Category:"""
            inputs = tokenizer(prompt, return_tensors="pt", max_length=800, truncation=True)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=10, temperature=0, do_sample=False)
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
            predictions.append(extract_label(response))
        
        accuracy = accuracy_score(test_df['Label'], predictions)
        results['few_shot'] = accuracy
        print(f"Few-shot Accuracy: {accuracy:.3f}")
        
        # 3. CHAIN-OF-THOUGHT
        print("\n3. Running Chain-of-thought...")
        predictions = []
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="CoT"):
            prompt = f"""Think step-by-step to classify this legal text.

Text: {row['Sentence'][:300]}

Reasoning: What is this text doing? Is it stating facts, asking legal questions, providing rules, analyzing, or concluding?
Category:"""
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=30, temperature=0, do_sample=False)
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
            predictions.append(extract_label(response))
        
        accuracy = accuracy_score(test_df['Label'], predictions)
        results['chain_of_thought'] = accuracy
        print(f"Chain-of-thought Accuracy: {accuracy:.3f}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        return results
        
    except Exception as e:
        print(f"Error with {model_name}: {e}")
        traceback.print_exc()
        
        # Clean up on error
        try:
            del model
        except:
            pass
        torch.cuda.empty_cache()
        gc.collect()
        
        return None

# Main execution
print("="*60)
print("LAMUS EXPERIMENTS - AS PER PROFESSOR")
print(f"Started: {datetime.now()}")
print("Models: Llama-3-8B, SaulLM-7B, Law-LLM")
print("Experiments: Zero-shot, Few-shot, Chain-of-thought")
print("="*60)

all_results = {}

for model_name, model_path in PROFESSOR_MODELS:  # Use PROFESSOR_MODELS, not MODELS!
    result = run_all_experiments(model_name, model_path)
    if result:
        all_results[model_name] = result
        
        # Save intermediate results
        with open(f'professor_results_{datetime.now().strftime("%Y%m%d")}.json', 'w') as f:
            json.dump(all_results, f, indent=2)
    
    time.sleep(10)

# Final summary
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
print(f"Completed: {datetime.now()}")
print(f"\nBaselines:")
print(f"  Random: 17.3%")
print(f"  Majority (Facts): 62.0%")
print(f"\nGemini (already tested): 6.5%")

for model, results in all_results.items():
    print(f"\n{model}:")
    print(f"  Zero-shot: {results.get('zero_shot', 0):.3f}")
    print(f"  Few-shot: {results.get('few_shot', 0):.3f}")
    print(f"  Chain-of-thought: {results.get('chain_of_thought', 0):.3f}")

print("\nResults saved to professor_results_*.json")