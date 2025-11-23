# Save as run_all_lamus_experiments.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import json
import time
from datetime import datetime
import gc
import os

# Your HuggingFace token
HF_TOKEN = 'hf_oxD'

# Load full dataset
print("Loading data...")
train_df = pd.read_csv('train_final.csv')
test_df = pd.read_csv('test_final.csv')  # Full 647 samples

print(f"Train: {len(train_df)} samples")
print(f"Test: {len(test_df)} samples")

# Valid labels
LABELS = ['Facts', 'Issue', 'Rule/Law/Holding', 'Analysis', 'Conclusion', 'Others']

class LAMUSModelTester:
    def __init__(self, model_name, model_path, token):
        self.model_name = model_name
        self.model_path = model_path
        self.token = token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
        
    def load_model(self):
        """Load model with appropriate settings"""
        print(f"Loading {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, token=self.token)
        
        # Different loading strategies based on model size
        if "54B" in self.model_path or "13b" in self.model_path.lower():
            print("Loading large model with 8-bit quantization...")
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                token=self.token,
                quantization_config=quantization_config,
                device_map="auto"
            )
        elif "t5" in self.model_path.lower():
            from transformers import AutoModelForSeq2SeqLM
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_path,
                token=self.token,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                token=self.token,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def create_zero_shot_prompt(self, sentence):
        """Zero-shot prompt"""
        return f"""Classify the following legal sentence into one of these categories:
Facts, Issue, Rule/Law/Holding, Analysis, Conclusion, Others

Sentence: "{sentence[:500]}"

Category:"""
    
    def create_few_shot_prompt(self, sentence, n_examples=2):
        """Few-shot prompt with examples"""
        examples = []
        for label in LABELS:
            samples = train_df[train_df['Label'] == label].sample(min(n_examples, 20), random_state=42)
            for _, row in samples.head(n_examples).iterrows():
                examples.append(f'Sentence: "{row["Sentence"][:150]}"\nCategory: {label}')
        
        prompt = "Classify legal sentences. Examples:\n\n"
        prompt += "\n\n".join(examples[:n_examples*2])  # Use 2 examples per class
        prompt += f"\n\nNow classify:\nSentence: \"{sentence[:500]}\"\nCategory:"
        
        return prompt
    
    def create_cot_prompt(self, sentence):
        """Chain-of-thought prompt"""
        return f"""Analyze and classify this legal sentence step by step.

Categories: Facts, Issue, Rule/Law/Holding, Analysis, Conclusion, Others

Sentence: "{sentence[:500]}"

Let me think:
1. What is the main content?
2. Is it factual, legal question, rule, analysis, conclusion, or procedural?
3. Which category fits best?

Category:"""
    
    def run_experiment(self, prompt_type="zero-shot", n_examples=2):
        """Run a single experiment"""
        print(f"\nRunning {prompt_type} experiment...")
        predictions = []
        
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=prompt_type):
            # Create prompt based on type
            if prompt_type == "zero-shot":
                prompt = self.create_zero_shot_prompt(row['Sentence'])
            elif prompt_type == "few-shot":
                prompt = self.create_few_shot_prompt(row['Sentence'], n_examples)
            elif prompt_type == "cot":
                prompt = self.create_cot_prompt(row['Sentence'])
            else:
                prompt = self.create_zero_shot_prompt(row['Sentence'])
            
            # Generate prediction
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=0.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=False
                )
            
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
            
            # Extract prediction
            pred = self.extract_label(response)
            predictions.append(pred)
            
            # Clear cache periodically
            if idx % 50 == 0:
                torch.cuda.empty_cache()
        
        # Calculate metrics
        accuracy = accuracy_score(test_df['Label'], predictions)
        report = classification_report(test_df['Label'], predictions, labels=LABELS, output_dict=True, zero_division=0)
        
        result = {
            'prompt_type': prompt_type,
            'accuracy': accuracy,
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score'],
            'predictions': predictions,
            'classification_report': report
        }
        
        self.results[prompt_type] = result
        
        print(f"{prompt_type} - Accuracy: {accuracy:.3f}, Macro F1: {report['macro avg']['f1-score']:.3f}")
        
        return result
    
    def extract_label(self, response):
        """Extract label from model response"""
        response = response.strip()
        
        # Check for exact matches first
        for label in LABELS:
            if label in response or label.lower() in response.lower():
                return label
        
        # Check first word
        first_word = response.split()[0] if response else ""
        for label in LABELS:
            if first_word.lower() in label.lower():
                return label
        
        # Default to Others
        return 'Others'
    
    def run_all_experiments(self):
        """Run all experiment types"""
        self.load_model()
        
        # Run experiments
        self.run_experiment("zero-shot")
        self.run_experiment("few-shot", n_examples=1)
        self.run_experiment("few-shot", n_examples=2)
        self.run_experiment("cot")
        
        # Save results
        self.save_results()
        
        # Clean up
        del self.model
        torch.cuda.empty_cache()
        gc.collect()
    
    def save_results(self):
        """Save all results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory
        os.makedirs("model_results", exist_ok=True)
        
        # Save detailed results
        filename = f"model_results/{self.model_name.replace('/', '_')}_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump({
                'model': self.model_name,
                'model_path': self.model_path,
                'test_samples': len(test_df),
                'results': {k: {kk: vv for kk, vv in v.items() if kk != 'predictions'} 
                          for k, v in self.results.items()}
            }, f, indent=2)
        
        print(f"Results saved to {filename}")

# Models to test (in priority order)
MODELS_TO_TEST = [
    # Legal models (highest priority)
    ("SaulLM-7B", "Equall/Saul-7B-Instruct-v1"),
    ("Law-LLM", "AdaptLLM/law-LLM"),
    
    # Medium-sized general models
    ("Llama-2-7B", "meta-llama/Llama-2-7b-chat-hf"),
    ("Mistral-7B", "mistralai/Mistral-7B-Instruct-v0.2"),
    
    # Smaller models (faster)
    ("Phi-3", "microsoft/Phi-3-mini-4k-instruct"),
    
    # Skip these for now (too large or slow)
    # ("SaulLM-54B", "Equall/SaulLM-54B-Instruct"),  # 54B - too large
    # ("Llama-3-8B", "meta-llama/Meta-Llama-3-8B-Instruct"),  # Already downloaded
]

def main():
    print("="*70)
    print("LAMUS COMPLETE EXPERIMENT RUNNER")
    print("="*70)
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Test samples: {len(test_df)}")
    print(f"Models to test: {len(MODELS_TO_TEST)}")
    print("="*70)
    
    all_results = {}
    
    for model_name, model_path in MODELS_TO_TEST:
        try:
            print(f"\n{'='*70}")
            print(f"TESTING: {model_name}")
            print('='*70)
            
            tester = LAMUSModelTester(model_name, model_path, HF_TOKEN)
            tester.run_all_experiments()
            
            # Store summary
            all_results[model_name] = {
                'zero_shot': tester.results.get('zero-shot', {}).get('accuracy', 0),
                'few_shot_1': tester.results.get('few-shot', {}).get('accuracy', 0),
                'cot': tester.results.get('cot', {}).get('accuracy', 0)
            }
            
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            continue
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY - ALL MODELS")
    print("="*70)
    
    for model, results in all_results.items():
        print(f"\n{model}:")
        for exp_type, acc in results.items():
            print(f"  {exp_type}: {acc:.3f}")
    
    # Save consolidated results
    with open(f"model_results/consolidated_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()