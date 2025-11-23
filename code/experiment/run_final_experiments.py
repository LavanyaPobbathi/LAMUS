# Save as run_lamus_working.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import json
from datetime import datetime

HF_TOKEN = 'hf_oxD'

# Load data
test_df = pd.read_csv('test_final.csv')
train_df = pd.read_csv('train_final.csv')

print(f"Running LAMUS experiments on {len(test_df)} test samples")
print("Models to test as per professor's requirements")

def test_model_simple(model_name, model_path):
    """Test a model with zero-shot only first"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print('='*60)
    
    try:
        # Use the default cache location
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            token=HF_TOKEN,
            cache_dir="/home/lavanya/.cache/huggingface"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            token=HF_TOKEN,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir="/home/lavanya/.cache/huggingface"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        predictions = []
        
        # Simple zero-shot test
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Classifying"):
            prompt = f"""Classify this legal sentence into one of these categories:
Facts, Issue, Rule/Law/Holding, Analysis, Conclusion, Others

Sentence: {row['Sentence'][:400]}

Answer with only the category name.
Category:"""
            
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).cuda()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
            
            # Extract prediction
            pred = "Others"
            response_clean = response.strip().lower()
            if "facts" in response_clean:
                pred = "Facts"
            elif "issue" in response_clean:
                pred = "Issue"
            elif "rule" in response_clean or "law" in response_clean or "holding" in response_clean:
                pred = "Rule/Law/Holding"
            elif "analysis" in response_clean:
                pred = "Analysis"
            elif "conclusion" in response_clean:
                pred = "Conclusion"
            
            predictions.append(pred)
        
        # Calculate accuracy
        accuracy = accuracy_score(test_df['Label'], predictions)
        report = classification_report(test_df['Label'], predictions)
        
        print(f"\nResults for {model_name}:")
        print(f"Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(report)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        with open(f'{model_name}_results_{timestamp}.txt', 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Test samples: {len(test_df)}\n")
            f.write(f"Accuracy: {accuracy:.3f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
        return accuracy
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

# Test with available models
results = {}

# Start with Llama-3 (we know it's downloaded)
acc = test_model_simple("Llama-3-8B", "meta-llama/Meta-Llama-3-8B-Instruct")
if acc:
    results["Llama-3-8B"] = acc

print("\n" + "="*60)
print("SUMMARY OF RESULTS")
print("="*60)
print(f"Baseline (Majority Class): 0.620")
print(f"Baseline (Random): 0.173")
for model, accuracy in results.items():
    print(f"{model}: {accuracy:.3f}")