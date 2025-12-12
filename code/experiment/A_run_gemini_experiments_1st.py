#!/usr/bin/env python3
"""
LAMUS Gemini Experiment Runner
==============================
Tests Gemini 2.5 Flash with all 3 prompts

Run with: nohup python3 run_gemini_experiments.py > gemini_log.txt 2>&1 &
"""

import os
import sys
import json
import time
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================
# UPDATE THIS WITH YOUR NEW GEMINI API KEY
# ============================================
GEMINI_API_KEY = "AIzaSyBFnBy2MHaY_atwTjCn8wcej-NND13JnBo"
# ============================================

# Labels
LABELS = ['Facts', 'Issue', 'Rule/Law/Holding', 'Analysis', 'Conclusion', 'Others']

# Label mapping
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
    """Extract classification label from model response"""
    if not response:
        return "Others"
    
    response = str(response).strip()
    
    import re
    label_match = re.search(r'Label:\s*(\S+)', response, re.IGNORECASE)
    if label_match:
        found_label = label_match.group(1).strip()
        found_label = re.sub(r'[^\w/]', '', found_label)
        if found_label in LABEL_MAP:
            return LABEL_MAP[found_label]
        for label in LABELS:
            if label.lower() == found_label.lower():
                return label
    
    response_lower = response.lower()
    
    for label in valid_labels:
        if label.lower() == response_lower:
            return label
        if response_lower.startswith(label.lower()):
            return label
    
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
    
    for label in valid_labels:
        if label.lower() in response_lower:
            return label
    
    return "Others"

def format_prompt_with_sentence(base_prompt, sentence):
    """Format prompt with the sentence to classify"""
    if "[INSERT SENTENCE HERE]" in base_prompt:
        return base_prompt.replace("[INSERT SENTENCE HERE]", sentence)
    elif "{sentence}" in base_prompt:
        return base_prompt.format(sentence=sentence)
    else:
        return f"{base_prompt}\n\nSentence: \"{sentence}\""

# ============================================================================
# GEMINI MODEL CLASS
# ============================================================================

class GeminiModel:
    """Gemini 2.5 Flash API wrapper"""
    
    def __init__(self):
        import google.generativeai as genai
        
        print("  Initializing Gemini 2.5 Flash...")
        self.name = "Gemini-2.5-Flash"
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.genai = genai
        print("  ✅ Gemini initialized!")
    
    def generate(self, prompt, max_new_tokens=100):
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.genai.GenerationConfig(
                    max_output_tokens=max_new_tokens,
                    temperature=0
                )
            )
            return response.text.strip()
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                print(f"\n  ⚠️ Rate limit hit, waiting 30 seconds...")
                time.sleep(30)
                # Retry once
                try:
                    response = self.model.generate_content(
                        prompt,
                        generation_config=self.genai.GenerationConfig(
                            max_output_tokens=max_new_tokens,
                            temperature=0
                        )
                    )
                    return response.text.strip()
                except:
                    return ""
            else:
                print(f"\n  Error: {error_msg[:50]}")
                return ""
    
    def cleanup(self):
        pass

# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_experiment(model, model_name, base_prompt, prompt_name, test_df):
    """Run a single experiment"""
    
    print(f"\n{'='*60}")
    print(f"Running: {model_name} with {prompt_name}")
    print(f"Test samples: {len(test_df)}")
    print(f"{'='*60}")
    
    predictions = []
    responses_log = []
    true_labels = test_df['Label'].tolist()
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"{model_name}-{prompt_name}"):
        sentence = row['Sentence'][:500]
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
        
        # Rate limiting - small delay between requests
        time.sleep(0.3)
    
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
        "responses_sample": responses_log[:10],
        "timestamp": datetime.now().isoformat()
    }
    
    print(f"\n✅ {model_name} - {prompt_name}: {accuracy*100:.2f}% accuracy")
    
    print(f"   Per-class F1:")
    for label in LABELS:
        if label in report:
            f1 = report[label]['f1-score']
            print(f"     {label}: {f1*100:.1f}%")
    
    return result


def main():
    print("="*70)
    print("LAMUS GEMINI EXPERIMENT RUNNER")
    print(f"Started: {datetime.now()}")
    print("="*70)
    
    # Load prompts
    print("\n📄 Loading Serene's prompts from files...")
    prompts = load_prompts()
    
    # Load data
    print("\n📊 Loading data...")
    train_df = pd.read_csv('train_final.csv')
    test_df = pd.read_csv('test_final.csv')
    print(f"  Train: {len(train_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    # Baseline
    baseline = (test_df['Label'] == 'Facts').mean()
    print(f"\n📈 Baseline (majority class 'Facts'): {baseline*100:.2f}%")
    
    print(f"\n🤖 Model: Gemini-2.5-Flash")
    print(f"📝 Prompts: {list(prompts.keys())}")
    print(f"🧪 Total experiments: 3")
    
    print("\n🚀 Starting Gemini experiments...")
    
    all_results = []
    
    try:
        # Initialize Gemini
        model = GeminiModel()
        
        # Run all prompts
        for prompt_name, base_prompt in prompts.items():
            try:
                result = run_experiment(
                    model=model,
                    model_name="Gemini-2.5-Flash",
                    base_prompt=base_prompt,
                    prompt_name=prompt_name,
                    test_df=test_df
                )
                all_results.append(result)
                
                # Save after each experiment
                with open('gemini_results_intermediate.json', 'w') as f:
                    json.dump(all_results, f, indent=2, default=str)
                print(f"  💾 Saved intermediate results")
                
            except Exception as e:
                print(f"\n❌ Error in {prompt_name}: {e}")
                all_results.append({
                    "model": "Gemini-2.5-Flash",
                    "prompt": prompt_name,
                    "accuracy": None,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        model.cleanup()
        
    except Exception as e:
        print(f"\n❌ Failed to initialize Gemini: {e}")
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'gemini_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Summary
    print("\n" + "="*70)
    print("📊 GEMINI EXPERIMENT SUMMARY")
    print("="*70)
    print(f"\nBaseline: {baseline*100:.2f}%")
    print(f"\n{'Prompt':<20} | {'Accuracy':>10} | {'vs Baseline':>12}")
    print("-"*50)
    
    for result in all_results:
        if result.get('accuracy') is not None:
            acc = result['accuracy'] * 100
            diff = acc - (baseline * 100)
            status = "✅" if diff > 0 else "⚠️"
            print(f"{status} {result['prompt']:<18} | {acc:>9.2f}% | {diff:>+11.2f}%")
        else:
            print(f"❌ {result['prompt']:<18} | {'ERROR':>10} |")
    
    # Best result
    valid_results = [r for r in all_results if r.get('accuracy') is not None]
    if valid_results:
        best = max(valid_results, key=lambda x: x['accuracy'])
        print(f"\n🏆 Best Gemini Result: {best['prompt']} with {best['accuracy']*100:.2f}%")
    
    print(f"\n💾 Results saved to: {results_file}")
    print(f"⏱️  Completed: {datetime.now()}")
    print("="*70)


if __name__ == "__main__":
    main()