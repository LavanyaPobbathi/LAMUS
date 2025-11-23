"""
HuggingFace Inference API Implementation for LAMUS
Uses HuggingFace API for all models without local downloads
"""

import pandas as pd
import numpy as np
import requests
import json
import time
from typing import List, Dict
from tqdm import tqdm
import logging
from lamus_experiment_runner import LAMUSExperimentRunner

logging.basicConfig(level=logging.INFO)

class HuggingFaceAPIClassifier(LAMUSExperimentRunner):
    """HuggingFace API implementation for all models"""
    
    # Model configurations
    MODEL_CONFIGS = {
        # General Domain Models
        'Llama-3-8B': {
            'api_name': 'meta-llama/Meta-Llama-3-8B-Instruct',
            'requires_access': True,
            'max_tokens': 2048,
            'type': 'general'
        },
        'Llama-3.1-8B': {
            'api_name': 'meta-llama/Llama-3.1-8B-Instruct',
            'requires_access': True,
            'max_tokens': 2048,
            'type': 'general'
        },
        'Qwen2.5-7B': {
            'api_name': 'Qwen/Qwen2.5-7B-Instruct',
            'requires_access': False,
            'max_tokens': 2048,
            'type': 'general'
        },
        # Legal Domain Models
        'SaulLM-7B': {
            'api_name': 'Equall/Saul-7B-Instruct-v1',
            'requires_access': False,
            'max_tokens': 2048,
            'type': 'legal'
        },
        'SaulLM-54B': {
            'api_name': 'Equall/SaulLM-54B-Instruct',
            'requires_access': True,  # May require access request
            'max_tokens': 2048,
            'type': 'legal'
        },
        'Law-LLM-7B': {
            'api_name': 'AdaptLLM/law-LLM',
            'requires_access': False,
            'max_tokens': 2048,
            'type': 'legal'
        }
    }
    
    def __init__(self, train_path: str, test_path: str, hf_token: str, model_name: str):
        """Initialize with HuggingFace API token and model selection"""
        super().__init__(train_path, test_path)
        
        self.hf_token = hf_token
        self.model_name = model_name
        
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.MODEL_CONFIGS.keys())}")
        
        self.model_config = self.MODEL_CONFIGS[model_name]
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_config['api_name']}"
        
        # Rate limiting
        self.requests_per_minute = 20  # Conservative rate limit
        self.last_request_time = 0
        self.retry_count = 3
        self.retry_delay = 5
        
        logging.info(f"Initialized {model_name} ({self.model_config['type']} domain)")
        logging.info(f"API URL: {self.api_url}")
        
        if self.model_config['requires_access']:
            logging.warning(f"⚠️  {model_name} may require access approval on HuggingFace")
            logging.warning(f"   Request access at: https://huggingface.co/{self.model_config['api_name']}")
    
    def rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 60.0 / self.requests_per_minute
        
        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def format_prompt_for_model(self, prompt: str) -> str:
        """Format prompt based on model type"""
        if 'Llama' in self.model_name:
            # Llama format
            formatted = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a legal expert classifier.<|eot_id|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        elif 'Saul' in self.model_name:
            # SaulLM format
            formatted = f"<|im_start|>system\nYou are a legal expert specializing in analyzing judicial opinions.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant"
        elif 'Qwen' in self.model_name:
            # Qwen format
            formatted = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant"
        else:
            # Default format
            formatted = f"System: You are a legal expert classifier.\n\nUser: {prompt}\n\nAssistant:"
        
        return formatted
    
    def call_hf_api(self, prompt: str, temperature: float = 0.1) -> str:
        """Call HuggingFace Inference API"""
        
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json"
        }
        
        # Format prompt for specific model
        formatted_prompt = self.format_prompt_for_model(prompt)
        
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": 50,
                "temperature": temperature,
                "do_sample": temperature > 0,
                "return_full_text": False,
                "stop": ["\n", ".", ","]
            },
            "options": {
                "wait_for_model": True,
                "use_cache": False
            }
        }
        
        for attempt in range(self.retry_count):
            try:
                self.rate_limit()
                
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract the generated text
                    if isinstance(result, list) and len(result) > 0:
                        generated_text = result[0].get('generated_text', '').strip()
                    else:
                        generated_text = str(result).strip()
                    
                    # Clean and extract just the label
                    if generated_text:
                        # Take first line/word as the label
                        label = generated_text.split('\n')[0].split()[0] if generated_text else "Others"
                        return label
                    
                    return "Others"
                
                elif response.status_code == 503:
                    # Model is loading
                    wait_time = min(30, self.retry_delay * (attempt + 1))
                    logging.info(f"Model is loading... waiting {wait_time} seconds")
                    time.sleep(wait_time)
                    continue
                    
                elif response.status_code == 403:
                    logging.error(f"Access denied to model. Please request access at: https://huggingface.co/{self.model_config['api_name']}")
                    return "Others"
                    
                else:
                    logging.error(f"API error {response.status_code}: {response.text}")
                    if attempt < self.retry_count - 1:
                        time.sleep(self.retry_delay)
                        continue
                    return "Others"
                    
            except Exception as e:
                logging.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)
                    continue
                return "Others"
        
        return "Others"
    
    def test_api_connection(self) -> bool:
        """Test if API connection works"""
        logging.info(f"Testing API connection for {self.model_name}...")
        
        test_prompt = "Classify this sentence: The defendant was found guilty. Respond with only: Facts, Issue, Rule/Law/Holding, Analysis, Conclusion, or Others."
        result = self.call_hf_api(test_prompt)
        
        if result and result != "Others":
            logging.info(f"✓ API test successful! Response: {result}")
            return True
        else:
            logging.error(f"✗ API test failed. Check your token and model access.")
            return False
    
    def run_zero_shot_experiment(self, sample_size: int = None):
        """Run zero-shot classification"""
        logging.info(f"Starting {self.model_name} zero-shot experiment...")
        
        # Test API first
        if not self.test_api_connection():
            logging.error("API test failed. Please check your configuration.")
            return None
        
        test_data = self.test_df.sample(n=sample_size, random_state=42) if sample_size else self.test_df
        
        predictions = []
        true_labels = test_data['Label'].tolist()
        
        for idx, row in tqdm(test_data.iterrows(), total=len(test_data),
                            desc=f"Zero-shot {self.model_name}"):
            prompt = self.create_zero_shot_prompt(row['Sentence'])
            prediction = self.call_hf_api(prompt, temperature=0.0)
            predictions.append(prediction)
            
            if len(predictions) % 10 == 0:
                logging.info(f"Processed {len(predictions)}/{len(test_data)} samples")
        
        # Evaluate
        result = self.evaluate_predictions(true_labels, predictions,
                                          self.model_name, "zero-shot")
        self.results[f'{self.model_name}_zero_shot'] = result
        
        return result
    
    def run_few_shot_experiment(self, sample_size: int = None, n_examples: int = 2):
        """Run few-shot classification"""
        logging.info(f"Starting {self.model_name} few-shot experiment with {n_examples} examples...")
        
        test_data = self.test_df.sample(n=sample_size, random_state=42) if sample_size else self.test_df
        
        predictions = []
        true_labels = test_data['Label'].tolist()
        
        for idx, row in tqdm(test_data.iterrows(), total=len(test_data),
                            desc=f"Few-shot {self.model_name}"):
            prompt = self.create_few_shot_prompt(row['Sentence'], n_examples)
            prediction = self.call_hf_api(prompt, temperature=0.0)
            predictions.append(prediction)
            
            if len(predictions) % 10 == 0:
                logging.info(f"Processed {len(predictions)}/{len(test_data)} samples")
        
        # Evaluate
        result = self.evaluate_predictions(true_labels, predictions,
                                          self.model_name, f"few-shot-{n_examples}")
        self.results[f'{self.model_name}_few_shot_{n_examples}'] = result
        
        return result
    
    def run_chain_of_thought_experiment(self, sample_size: int = None):
        """Run chain-of-thought classification"""
        logging.info(f"Starting {self.model_name} chain-of-thought experiment...")
        
        test_data = self.test_df.sample(n=sample_size, random_state=42) if sample_size else self.test_df
        
        predictions = []
        true_labels = test_data['Label'].tolist()
        
        for idx, row in tqdm(test_data.iterrows(), total=len(test_data),
                            desc=f"CoT {self.model_name}"):
            prompt = self.create_chain_of_thought_prompt(row['Sentence'])
            prediction = self.call_hf_api(prompt, temperature=0.1)
            predictions.append(prediction)
            
            if len(predictions) % 10 == 0:
                logging.info(f"Processed {len(predictions)}/{len(test_data)} samples")
        
        # Evaluate
        result = self.evaluate_predictions(true_labels, predictions,
                                          self.model_name, "chain-of-thought")
        self.results[f'{self.model_name}_cot'] = result
        
        return result
    
    def run_all_experiments(self, sample_size: int = None):
        """Run all experiments for this model"""
        print("\n" + "="*60)
        print(f"{self.model_name.upper()} EXPERIMENTS (via HF API)")
        print("="*60)
        
        results = {}
        
        # Zero-shot
        result = self.run_zero_shot_experiment(sample_size)
        if result:
            results['zero_shot'] = result
            print(f"✓ Zero-shot complete - F1: {result['macro_f1']*100:.1f}%")
        
        # Few-shot
        result = self.run_few_shot_experiment(sample_size, n_examples=2)
        if result:
            results['few_shot'] = result
            print(f"✓ Few-shot complete - F1: {result['macro_f1']*100:.1f}%")
        
        # Chain-of-thought
        result = self.run_chain_of_thought_experiment(sample_size)
        if result:
            results['cot'] = result
            print(f"✓ Chain-of-thought complete - F1: {result['macro_f1']*100:.1f}%")
        
        # Save results
        self.save_results(f"{self.model_name}_results")
        
        return results


def run_all_models(hf_token: str, sample_size: int = None):
    """Run experiments for all available models"""
    
    # Models to test (in priority order)
    models_to_test = [
        # Legal domain (priority)
        'SaulLM-7B',
        'Law-LLM-7B',
        # General domain
        'Qwen2.5-7B',
        'Llama-3.1-8B',
        # Large models (if accessible)
        'SaulLM-54B',
    ]
    
    all_results = {}
    
    for model_name in models_to_test:
        print(f"\n{'='*60}")
        print(f"Testing {model_name}")
        print('='*60)
        
        try:
            classifier = HuggingFaceAPIClassifier(
                train_path="train_final.csv",
                test_path="test_final.csv",
                hf_token=hf_token,
                model_name=model_name
            )
            
            # Run experiments
            results = classifier.run_all_experiments(sample_size)
            all_results[model_name] = results
            
        except Exception as e:
            print(f"❌ Failed to run {model_name}: {e}")
            continue
    
    return all_results


def main():
    """Main execution"""
    import sys
    import getpass
    
    print("="*60)
    print("HUGGINGFACE API - LAMUS EXPERIMENTS")
    print("="*60)
    
    print("\n📋 Prerequisites:")
    print("1. HuggingFace account: https://huggingface.co/join")
    print("2. API token: https://huggingface.co/settings/tokens")
    print("3. Request access to gated models:")
    print("   - Llama-3: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct")
    print("   - SaulLM-54B: https://huggingface.co/Equall/SaulLM-54B-Instruct")
    
    # Get API token
    print("\n🔑 Enter your HuggingFace API token")
    print("   (Find it at: https://huggingface.co/settings/tokens)")
    hf_token = getpass.getpass("HF Token: ").strip()
    
    if not hf_token:
        print("❌ API token required!")
        sys.exit(1)
    
    # Model selection
    print("\n📊 Select experiment mode:")
    print("1. Test single model")
    print("2. Run all accessible models")
    print("3. Run only legal domain models")
    print("4. Run only general domain models")
    
    choice = input("\nChoice (1-4): ").strip()
    
    # Sample size
    sample_input = input("\n📏 Sample size for testing (or Enter for full dataset): ").strip()
    sample_size = int(sample_input) if sample_input else None
    
    if sample_size:
        print(f"\n⚠️  Running on {sample_size} samples (testing mode)")
    else:
        print(f"\n📊 Running on full dataset")
        confirm = input("This will take time and API calls. Continue? (y/n): ")
        if confirm.lower() != 'y':
            sys.exit(0)
    
    if choice == '1':
        # Single model
        print("\nAvailable models:")
        models = list(HuggingFaceAPIClassifier.MODEL_CONFIGS.keys())
        for i, model in enumerate(models, 1):
            config = HuggingFaceAPIClassifier.MODEL_CONFIGS[model]
            access = "🔒 Requires Access" if config['requires_access'] else "✓ Open"
            print(f"{i}. {model} ({config['type']}) - {access}")
        
        model_idx = int(input("\nSelect model (number): ")) - 1
        model_name = models[model_idx]
        
        classifier = HuggingFaceAPIClassifier(
            train_path="train__2_.csv",
            test_path="shuffledtest.csv",
            hf_token=hf_token,
            model_name=model_name
        )
        
        classifier.run_all_experiments(sample_size)
        
    elif choice == '2':
        # All models
        run_all_models(hf_token, sample_size)
        
    elif choice == '3':
        # Legal models only
        legal_models = ['SaulLM-7B', 'Law-LLM-7B', 'SaulLM-54B']
        for model_name in legal_models:
            try:
                classifier = HuggingFaceAPIClassifier(
                    train_path="train__2_.csv",
                    test_path="shuffledtest.csv",
                    hf_token=hf_token,
                    model_name=model_name
                )
                classifier.run_all_experiments(sample_size)
            except Exception as e:
                print(f"Failed {model_name}: {e}")
                
    elif choice == '4':
        # General models only
        general_models = ['Qwen2.5-7B', 'Llama-3.1-8B']
        for model_name in general_models:
            try:
                classifier = HuggingFaceAPIClassifier(
                    train_path="train__2_.csv",
                    test_path="shuffledtest.csv",
                    hf_token=hf_token,
                    model_name=model_name
                )
                classifier.run_all_experiments(sample_size)
            except Exception as e:
                print(f"Failed {model_name}: {e}")
    
    print("\n✅ Experiments complete!")
    print("Run consolidate_results.py to generate final report")


if __name__ == "__main__":
    main()
