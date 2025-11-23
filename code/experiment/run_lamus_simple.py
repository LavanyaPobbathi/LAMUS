# Save as run_lamus_simple.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import json
from datetime import datetime

# Configuration
HF_TOKEN = 'hf_oxD'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# Load data
print("Loading data...")
test_df = pd.read_csv('test_final.csv')
train_df = pd.read_csv('train_final.csv')
print(f"Test samples: {len(test_df)}")

LABELS = ['Facts', 'Issue', 'Rule/Law/Holding', 'Analysis', 'Conclusion', 'Others']

def run_model_test(model_name, model_path, test_samples=100):
    """Run a quick test with a model"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print('='*60)
    
    try:
        # Load model and tokenizer
        print("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN)
        
        # Load with appropriate dtype
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            token=HF_TOKEN,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Test on subset
        test_subset = test_df.head(test_samples)
        predictions = []
        
        print(f"Running predictions on {test_samples} samples...")
        for idx, row in tqdm(test_subset.iterrows(), total=len(test_subset)):
            prompt = f"""Classify this legal sentence into ONE category:
Facts, Issue, Rule/Law/Holding, Analysis, Conclusion, or Others

Sentence: "{row['Sentence'][:400]}"

Answer with one word only. Category:"""
            
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=False
                )
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
            
            # Extract label
            pred = response.strip().split()[0] if response else "Others"
            # Validate label
            if pred not in LABELS:
                pred = "Others"
            predictions.append(pred)
        
        # Calculate metrics
        accuracy = accuracy_score(test_subset['Label'], predictions)
        print(f"\nAccuracy: {accuracy:.3f}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
        return accuracy
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

# Priority models
models_to_test = [
    # Start with model that's already downloaded
    ("Llama-3-8B", "meta-llama/Meta-Llama-3-8B-Instruct"),
    
    # Then legal models
    ("SaulLM-7B", "Equall/Saul-7B-Instruct-v1"),
    ("Law-LLM", "AdaptLLM/law-LLM"),
]

results = {}

for name, path in models_to_test:
    acc = run_model_test(name, path, test_samples=100)  # Test on 100 samples first
    if acc is not None:
        results[name] = acc
        print(f"{name}: {acc:.3f}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for model, acc in results.items():
    print(f"{model}: {acc:.3f}")