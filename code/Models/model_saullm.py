"""
SaulLM Implementation for LAMUS
Legal-specific language models for argument classification
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict
import time
import json
import os
from tqdm import tqdm
import logging
from lamus_experiment_runner import LAMUSExperimentRunner

logging.basicConfig(level=logging.INFO)

class SaulLMClassifier(LAMUSExperimentRunner):
    """SaulLM implementation for LAMUS experiments"""
    
    def __init__(self, train_path: str, test_path: str, model_name: str = "Equall/Saul-7B-Instruct-v1"):
        """Initialize with SaulLM model"""
        super().__init__(train_path, test_path)
        
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logging.info(f"Loading {model_name} on {self.device}...")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # For 7B model, use appropriate settings
        if "7B" in model_name:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
        else:
            # For 54B model, need more aggressive memory optimization
            logging.warning("54B model requires significant resources. Consider using API or cloud GPU.")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=True  # Use 8-bit quantization for large model
            )
        
        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logging.info(f"Model loaded successfully on {self.device}")
    
    def format_instruction(self, prompt: str) -> str:
        """Format prompt in SaulLM instruction format"""
        # SaulLM uses specific instruction formatting
        instruction = f"""<|im_start|>system
You are a legal expert specializing in analyzing judicial opinions and legal texts.
<|im_end|>
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
"""
        return instruction
    
    def get_model_prediction(self, prompt: str, max_new_tokens: int = 20, temperature: float = 0.1) -> str:
        """Get prediction from SaulLM model"""
        try:
            # Format the prompt
            formatted_prompt = self.format_instruction(prompt)
            
            # Tokenize
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                max_length=2048,
                truncation=True
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=False if temperature == 0 else True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and extract prediction
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            if "<|im_start|>assistant" in generated_text:
                response = generated_text.split("<|im_start|>assistant")[-1].strip()
            else:
                response = generated_text[len(formatted_prompt):].strip()
            
            # Clean response to get just the label
            response = response.split('\n')[0].strip()
            
            return response
            
        except Exception as e:
            logging.error(f"Model prediction error: {e}")
            return "Others"
    
    def run_zero_shot_experiment(self, sample_size: int = None):
        """Run zero-shot classification with SaulLM"""
        logging.info(f"Starting {self.model_name} zero-shot experiment...")
        
        test_data = self.test_df.sample(n=sample_size) if sample_size else self.test_df
        
        predictions = []
        true_labels = test_data['Label'].tolist()
        
        for idx, row in tqdm(test_data.iterrows(), total=len(test_data),
                            desc="Zero-shot predictions"):
            prompt = self.create_zero_shot_prompt(row['Sentence'])
            prediction = self.get_model_prediction(prompt, temperature=0.0)
            predictions.append(prediction)
            
            if len(predictions) % 10 == 0:
                logging.info(f"Processed {len(predictions)}/{len(test_data)} samples")
                # Clear cache periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Evaluate
        model_short_name = self.model_name.split('/')[-1]
        result = self.evaluate_predictions(true_labels, predictions,
                                          model_short_name, "zero-shot")
        self.results[f'{model_short_name}_zero_shot'] = result
        
        return result
    
    def run_few_shot_experiment(self, sample_size: int = None, n_examples: int = 2):
        """Run few-shot classification with SaulLM"""
        logging.info(f"Starting {self.model_name} few-shot experiment with {n_examples} examples...")
        
        test_data = self.test_df.sample(n=sample_size) if sample_size else self.test_df
        
        predictions = []
        true_labels = test_data['Label'].tolist()
        
        for idx, row in tqdm(test_data.iterrows(), total=len(test_data),
                            desc="Few-shot predictions"):
            prompt = self.create_few_shot_prompt(row['Sentence'], n_examples)
            prediction = self.get_model_prediction(prompt, temperature=0.0)
            predictions.append(prediction)
            
            if len(predictions) % 10 == 0:
                logging.info(f"Processed {len(predictions)}/{len(test_data)} samples")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Evaluate
        model_short_name = self.model_name.split('/')[-1]
        result = self.evaluate_predictions(true_labels, predictions,
                                          model_short_name, f"few-shot-{n_examples}")
        self.results[f'{model_short_name}_few_shot_{n_examples}'] = result
        
        return result
    
    def run_chain_of_thought_experiment(self, sample_size: int = None):
        """Run chain-of-thought classification with SaulLM"""
        logging.info(f"Starting {self.model_name} chain-of-thought experiment...")
        
        test_data = self.test_df.sample(n=sample_size) if sample_size else self.test_df
        
        predictions = []
        true_labels = test_data['Label'].tolist()
        
        for idx, row in tqdm(test_data.iterrows(), total=len(test_data),
                            desc="Chain-of-thought predictions"):
            prompt = self.create_chain_of_thought_prompt(row['Sentence'])
            prediction = self.get_model_prediction(prompt, max_new_tokens=100, temperature=0.1)
            
            # Extract final answer from CoT response
            if "Final answer:" in prediction:
                prediction = prediction.split("Final answer:")[-1].strip()
            elif "Category:" in prediction:
                prediction = prediction.split("Category:")[-1].strip()
            
            # Get first line/word as the label
            prediction = prediction.split('\n')[0].split()[0] if prediction else "Others"
            
            predictions.append(prediction)
            
            if len(predictions) % 10 == 0:
                logging.info(f"Processed {len(predictions)}/{len(test_data)} samples")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Evaluate
        model_short_name = self.model_name.split('/')[-1]
        result = self.evaluate_predictions(true_labels, predictions,
                                          model_short_name, "chain-of-thought")
        self.results[f'{model_short_name}_cot'] = result
        
        return result
    
    def run_all_experiments(self, sample_size: int = None):
        """Run all SaulLM experiments"""
        model_short_name = self.model_name.split('/')[-1]
        
        print("\n" + "="*60)
        print(f"{model_short_name.upper()} EXPERIMENTS")
        print("="*60)
        
        # Run experiments
        self.run_zero_shot_experiment(sample_size)
        print("✓ Zero-shot complete")
        
        self.run_few_shot_experiment(sample_size, n_examples=1)
        print("✓ 1-shot complete")
        
        self.run_few_shot_experiment(sample_size, n_examples=2)
        print("✓ 2-shot complete")
        
        self.run_chain_of_thought_experiment(sample_size)
        print("✓ Chain-of-thought complete")
        
        # Save results
        self.save_results(f"{model_short_name}_results")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return self.results


def main():
    """Main execution"""
    import sys
    
    print("="*60)
    print("SAULLM - LAMUS EXPERIMENTS")
    print("="*60)
    
    # Select model
    print("\nAvailable models:")
    print("1. Saul-7B-Instruct-v1 (Recommended - fits on most GPUs)")
    print("2. SaulLM-54B-Instruct (Requires high-end GPU or quantization)")
    print("3. Custom model path")
    
    choice = input("\nSelect model (1-3): ").strip()
    
    if choice == "1":
        model_name = "Equall/Saul-7B-Instruct-v1"
    elif choice == "2":
        model_name = "Equall/SaulLM-54B-Instruct"
        print("\n⚠️  Warning: 54B model requires significant GPU memory (>40GB)")
        confirm = input("Continue? (y/n): ")
        if confirm.lower() != 'y':
            sys.exit(0)
    elif choice == "3":
        model_name = input("Enter model name/path: ").strip()
    else:
        print("Invalid choice")
        sys.exit(1)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("\n⚠️  No GPU detected. Running on CPU will be very slow.")
        confirm = input("Continue anyway? (y/n): ")
        if confirm.lower() != 'y':
            sys.exit(0)
    else:
        print(f"\n✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Initialize classifier
    classifier = SaulLMClassifier(
        train_path="train__2_.csv",
        test_path="shuffledtest.csv",
        model_name=model_name
    )
    
    # Ask for sample size
    sample_input = input("\nEnter sample size for testing (or press Enter for full dataset): ").strip()
    sample_size = int(sample_input) if sample_input else None
    
    if sample_size:
        print(f"\n⚠️  Running experiments on {sample_size} samples (testing mode)")
    else:
        print(f"\n📊 Running experiments on full test set ({len(classifier.test_df)} samples)")
        estimated_time = len(classifier.test_df) * 2 / 60  # ~2 seconds per sample
        print(f"   Estimated time: {estimated_time:.1f} minutes")
        confirm = input("Continue? (y/n): ")
        if confirm.lower() != 'y':
            sys.exit(0)
    
    # Run all experiments
    results = classifier.run_all_experiments(sample_size)
    
    print("\n✅ All SaulLM experiments complete!")
    print(f"Results saved to {model_name.split('/')[-1]}_results/")


if __name__ == "__main__":
    main()
