#!/usr/bin/env python3
"""
LAMUS - Few-Shot Example Count Experiment
==========================================
Tests how the NUMBER of few-shot examples affects performance.

Professor Serene's Request:
"Could you use the prompts on the same models to test how many examples 
in the few shot prompts is most effective?"

Experiment Design:
- Few-shot with 1 example
- Few-shot with 3 examples
- Few-shot with 4 examples
- Few-shot with 5 examples

Models to test:
- Llama-3-8B (best performer)
- SaulLM-7B (legal model)
- law-LLM (legal model)
- Qwen3-Thinking (reasoning model)

Run: CUDA_VISIBLE_DEVICES=1 nohup python3 G_fewshot_examples_experiment.py > fewshot_examples.log 2>&1 &
Monitor: tail -f fewshot_examples.log

Time estimate: ~4-6 hours (4 models × 4 prompts × ~20 min each)
"""

import os
import sys
import gc
import json
import time
import torch
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:64')
os.environ.setdefault('HF_HOME', '/home/lavanya/.cache/huggingface')

# ============================================
# CONFIGURATION
# ============================================
HF_TOKEN = "HF_TOKEN_KEY"
OUTPUT_DIR = "./fewshot_examples_results"
MAX_LENGTH = 2048  # Longer for few-shot prompts
MAX_NEW_TOKENS = 50

# Models to test (all models that support prompting)
# NOTE: LegalBERT is NOT included because it's an encoder-only model
#       that cannot do prompting - only fine-tuning
MODELS = {
    "Llama-3-8B": "meta-llama/Meta-Llama-3-8B-Instruct",
    "SaulLM-54B": "Equall/Saul-Instruct-v1",  # 54B legal model
    "SaulLM-7B": "Equall/Saul-7B-Instruct-v1",
    "law-LLM": "AdaptLLM/law-LLM",
    "Qwen3-Thinking": "Qwen/Qwen2.5-7B-Instruct",
}

LABELS = ['Facts', 'Issue', 'Rule/Law/Holding', 'Analysis', 'Conclusion', 'Others']

# Directory containing few-shot prompt files
PROMPTS_DIR = "./F_Prompts"  # Change this to your prompts directory
# ============================================


def load_fewshot_prompts(prompts_dir):
    """
    Load few-shot prompts from .txt files in the specified directory.
    
    Expected file naming convention:
    - Few_Shot_(1_ex).txt or Few_Shot_1_ex.txt -> 1 example
    - Few_Shot_(3_ex).txt or Few_Shot_3_ex.txt -> 3 examples
    - etc.
    
    Returns dict: {prompt_name: prompt_text}
    """
    import glob
    import re
    
    prompts = {}
    
    # Find all .txt files that match few-shot pattern
    patterns = [
        os.path.join(prompts_dir, "Few_Shot*.txt"),
        os.path.join(prompts_dir, "Few Shot*.txt"),
        os.path.join(prompts_dir, "few_shot*.txt"),
        os.path.join(prompts_dir, "few-shot*.txt"),
    ]
    
    found_files = []
    for pattern in patterns:
        found_files.extend(glob.glob(pattern))
    
    # Remove duplicates
    found_files = list(set(found_files))
    
    if not found_files:
        print(f"⚠️ No few-shot prompt files found in {prompts_dir}")
        print("   Expected files like: Few_Shot_(1_ex).txt, Few_Shot_(3_ex).txt, etc.")
        return {}
    
    print(f"\n📂 Found {len(found_files)} prompt files:")
    
    for filepath in sorted(found_files):
        filename = os.path.basename(filepath)
        
        # Extract number of examples from filename
        # Matches patterns like: (1_ex), (3_ex), _1_ex, _3_ex, 1ex, 3ex
        match = re.search(r'[_\(]?(\d+)[_\s]?ex', filename, re.IGNORECASE)
        
        if match:
            num_examples = int(match.group(1))
            prompt_name = f"{num_examples}_example{'s' if num_examples > 1 else ''}"
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    prompt_text = f.read().strip()
                
                # Fix encoding issues (smart quotes)
                prompt_text = prompt_text.replace('â€™', "'")
                prompt_text = prompt_text.replace('â€œ', '"')
                prompt_text = prompt_text.replace('â€', '"')
                
                prompts[prompt_name] = prompt_text
                print(f"   ✅ {filename} -> {prompt_name} ({len(prompt_text)} chars)")
                
            except Exception as e:
                print(f"   ❌ Error reading {filename}: {e}")
        else:
            print(f"   ⚠️ Skipping {filename} - couldn't extract example count")
    
    return prompts


def get_num_examples_from_name(prompt_name):
    """Extract number of examples from prompt name like '3_examples' -> 3"""
    import re
    match = re.search(r'(\d+)', prompt_name)
    return int(match.group(1)) if match else 0


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


def load_model(model_name, model_path):
    """Load model with 4-bit quantization"""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    
    print(f"\n📥 Loading {model_name}...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        token=HF_TOKEN,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    
    print(f"   ✅ Loaded. GPU: {get_gpu_memory()}")
    return model, tokenizer


def extract_label(response):
    """Extract label from model response"""
    response_lower = response.lower()
    
    # Check for each label
    label_mapping = {
        'rule/law/holding': 'Rule/Law/Holding',
        'rule': 'Rule/Law/Holding',
        'law': 'Rule/Law/Holding',
        'holding': 'Rule/Law/Holding',
        'facts': 'Facts',
        'fact': 'Facts',
        'issue': 'Issue',
        'analysis': 'Analysis',
        'conclusion': 'Conclusion',
        'other': 'Others',
        'others': 'Others',
    }
    
    for key, label in label_mapping.items():
        if key in response_lower:
            return label
    
    return "Others"


def run_experiment(model, tokenizer, model_name, prompt_name, prompt_template, test_df):
    """Run a single few-shot experiment"""
    
    print(f"\n🔬 Testing: {model_name} with {prompt_name}")
    
    predictions = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"{model_name}-{prompt_name}"):
        sentence = str(row['Sentence'])[:500]  # Truncate long sentences
        
        # Build prompt
        full_prompt = f"""{prompt_template}

Now classify this sentence:
Sentence: "{sentence}"
Label:"""
        
        # Format for model
        if "Llama" in model_name or "llama" in model_name.lower():
            input_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{full_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            input_text = full_prompt
        
        try:
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
            pred = extract_label(response)
            
        except Exception as e:
            print(f"   ⚠️ Error at idx {idx}: {e}")
            pred = "Others"
        
        predictions.append(pred)
    
    # Calculate accuracy
    true_labels = test_df['Label'].tolist()
    accuracy = accuracy_score(true_labels, predictions)
    
    print(f"   ✅ Accuracy: {accuracy*100:.2f}%")
    
    return {
        'model': model_name,
        'prompt': prompt_name,
        'num_examples': get_num_examples_from_name(prompt_name),
        'accuracy': round(accuracy * 100, 2),
        'predictions': predictions,
    }


def main():
    print("="*70)
    print("LAMUS - FEW-SHOT EXAMPLE COUNT EXPERIMENT")
    print(f"Started: {datetime.now()}")
    print("="*70)
    print("\n📋 Professor Serene's Request:")
    print("   Test how many examples in few-shot prompts is most effective")
    
    if not torch.cuda.is_available():
        print("❌ No GPU!")
        return
    
    print(f"\n🖥️ GPU: {torch.cuda.get_device_name(0)}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load few-shot prompts from files
    print("\n📂 Loading few-shot prompts from files...")
    FEW_SHOT_PROMPTS = load_fewshot_prompts(PROMPTS_DIR)
    
    if not FEW_SHOT_PROMPTS:
        print("❌ No prompts loaded! Please check the prompts directory.")
        print(f"   Expected directory: {PROMPTS_DIR}")
        print("   Expected files: Few_Shot_(1_ex).txt, Few_Shot_(3_ex).txt, etc.")
        return
    
    print(f"\n📊 Experiment Design:")
    print(f"   - Prompts: {list(FEW_SHOT_PROMPTS.keys())}")
    print(f"   - Models: {list(MODELS.keys())}")
    print(f"   - Total experiments: {len(FEW_SHOT_PROMPTS) * len(MODELS)}")
    
    # Load test data
    print("\n📊 Loading test data...")
    test_df = pd.read_csv('test_final.csv')
    print(f"   Test samples: {len(test_df)}")
    
    # Results storage
    all_results = []
    results_file = os.path.join(OUTPUT_DIR, "fewshot_examples_results.json")
    
    # Load existing results if any
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            all_results = json.load(f)
        print(f"📂 Loaded {len(all_results)} existing results")
    
    # Track completed experiments
    completed = set()
    for r in all_results:
        completed.add((r['model'], r['prompt']))
    
    # Run experiments
    for model_name, model_path in MODELS.items():
        print(f"\n{'='*60}")
        print(f"📦 Model: {model_name}")
        print(f"{'='*60}")
        
        # Check if all prompts done for this model
        model_prompts_done = sum(1 for c in completed if c[0] == model_name)
        if model_prompts_done >= len(FEW_SHOT_PROMPTS):
            print(f"   ✅ All prompts already completed, skipping")
            continue
        
        # Load model
        clear_memory()
        try:
            model, tokenizer = load_model(model_name, model_path)
        except Exception as e:
            print(f"   ❌ Failed to load: {e}")
            continue
        
        # Test each prompt
        for prompt_name, prompt_template in FEW_SHOT_PROMPTS.items():
            if (model_name, prompt_name) in completed:
                print(f"   ⏭️ {prompt_name} already done, skipping")
                continue
            
            try:
                result = run_experiment(model, tokenizer, model_name, prompt_name, prompt_template, test_df)
                
                # Remove predictions for storage (too large)
                result_to_save = {k: v for k, v in result.items() if k != 'predictions'}
                all_results.append(result_to_save)
                completed.add((model_name, prompt_name))
                
                # Save after each experiment
                with open(results_file, 'w') as f:
                    json.dump(all_results, f, indent=2)
                print(f"   💾 Saved results")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Cleanup
        del model, tokenizer
        clear_memory()
    
    # Generate summary
    print("\n" + "="*70)
    print("📊 RESULTS SUMMARY")
    print("="*70)
    
    df_results = pd.DataFrame(all_results)
    
    if len(df_results) > 0:
        # Pivot table
        pivot = df_results.pivot(index='model', columns='num_examples', values='accuracy')
        print("\n📈 Accuracy by Model and Number of Examples:")
        print(pivot.to_string())
        
        # Best per model
        print("\n🏆 Best Configuration per Model:")
        for model in df_results['model'].unique():
            model_data = df_results[df_results['model'] == model]
            best = model_data.loc[model_data['accuracy'].idxmax()]
            print(f"   {model}: {best['num_examples']} examples -> {best['accuracy']}%")
        
        # Overall best
        best_overall = df_results.loc[df_results['accuracy'].idxmax()]
        print(f"\n🥇 Overall Best: {best_overall['model']} with {best_overall['num_examples']} examples -> {best_overall['accuracy']}%")
        
        # Save CSV
        csv_file = os.path.join(OUTPUT_DIR, "fewshot_examples_results.csv")
        df_results.to_csv(csv_file, index=False)
        print(f"\n💾 Saved: {csv_file}")
        
        # Compare with zero-shot baseline
        print("\n📊 Comparison with Zero-Shot (from previous experiments):")
        zero_shot_baselines = {
            'Llama-3-8B': 65.38,
            'SaulLM-54B': 67.39,
            'SaulLM-7B': 52.09,
            'law-LLM': 60.12,
            'Qwen3-Thinking': 56.11,
        }
        
        for model in df_results['model'].unique():
            if model in zero_shot_baselines:
                zs = zero_shot_baselines[model]
                model_data = df_results[df_results['model'] == model]
                best_fs = model_data['accuracy'].max()
                diff = best_fs - zs
                print(f"   {model}: Zero-Shot={zs}%, Best Few-Shot={best_fs}%, Change={diff:+.2f}%")
    
    print(f"\n✅ Experiment complete! {datetime.now()}")


if __name__ == "__main__":
    main()
