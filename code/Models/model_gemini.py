"""
Gemini 2.0 Flash Implementation for LAMUS
Using Google's Gemini API for legal argument classification
"""

import pandas as pd
import numpy as np
import google.generativeai as genai
from typing import List, Dict
import time
import json
import os
from tqdm import tqdm
import logging
from lamus_experiment_runner import LAMUSExperimentRunner

# Set up logging
logging.basicConfig(level=logging.INFO)

class GeminiClassifier(LAMUSExperimentRunner):
    """Gemini-specific implementation for LAMUS experiments"""
    
    def __init__(self, train_path: str, test_path: str, api_key: str):
        """Initialize with Gemini API"""
        super().__init__(train_path, test_path)
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Rate limiting
        self.requests_per_minute = 60
        self.last_request_time = 0
        
    def rate_limit(self):
        """Simple rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 60.0 / self.requests_per_minute
        
        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def get_gemini_prediction(self, prompt: str, temperature: float = 0.0) -> str:
        """Get prediction from Gemini"""
        try:
            self.rate_limit()
            
            generation_config = genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=50,
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Extract just the label from response
            prediction = response.text.strip()
            
            # Clean the prediction - get first line, remove extra text
            if '\n' in prediction:
                prediction = prediction.split('\n')[0]
            
            return prediction
            
        except Exception as e:
            logging.error(f"Gemini API error: {e}")
            return "Others"
    
    def run_zero_shot_experiment(self, sample_size: int = None):
        """Run zero-shot classification with Gemini"""
        logging.info("Starting Gemini zero-shot experiment...")
        
        # Use sample for testing or full dataset
        test_data = self.test_df.sample(n=sample_size) if sample_size else self.test_df
        
        predictions = []
        true_labels = test_data['Label'].tolist()
        
        for idx, row in tqdm(test_data.iterrows(), total=len(test_data), 
                            desc="Zero-shot predictions"):
            prompt = self.create_zero_shot_prompt(row['Sentence'])
            prediction = self.get_gemini_prediction(prompt)
            predictions.append(prediction)
            
            # Log every 10 predictions for monitoring
            if len(predictions) % 10 == 0:
                logging.info(f"Processed {len(predictions)}/{len(test_data)} samples")
        
        # Evaluate
        result = self.evaluate_predictions(true_labels, predictions, 
                                          "Gemini-2.0-Flash", "zero-shot")
        self.results['gemini_zero_shot'] = result
        
        return result
    
    def run_few_shot_experiment(self, sample_size: int = None, n_examples: int = 2):
        """Run few-shot classification with Gemini"""
        logging.info(f"Starting Gemini few-shot experiment with {n_examples} examples per class...")
        
        test_data = self.test_df.sample(n=sample_size) if sample_size else self.test_df
        
        predictions = []
        true_labels = test_data['Label'].tolist()
        
        for idx, row in tqdm(test_data.iterrows(), total=len(test_data),
                            desc="Few-shot predictions"):
            prompt = self.create_few_shot_prompt(row['Sentence'], n_examples)
            prediction = self.get_gemini_prediction(prompt)
            predictions.append(prediction)
            
            if len(predictions) % 10 == 0:
                logging.info(f"Processed {len(predictions)}/{len(test_data)} samples")
        
        # Evaluate
        result = self.evaluate_predictions(true_labels, predictions,
                                          "Gemini-2.0-Flash", f"few-shot-{n_examples}")
        self.results[f'gemini_few_shot_{n_examples}'] = result
        
        return result
    
    def run_chain_of_thought_experiment(self, sample_size: int = None):
        """Run chain-of-thought classification with Gemini"""
        logging.info("Starting Gemini chain-of-thought experiment...")
        
        test_data = self.test_df.sample(n=sample_size) if sample_size else self.test_df
        
        predictions = []
        true_labels = test_data['Label'].tolist()
        
        for idx, row in tqdm(test_data.iterrows(), total=len(test_data),
                            desc="Chain-of-thought predictions"):
            prompt = self.create_chain_of_thought_prompt(row['Sentence'])
            prediction = self.get_gemini_prediction(prompt, temperature=0.1)
            
            # Extract final answer from CoT response
            if "Final answer:" in prediction:
                prediction = prediction.split("Final answer:")[-1].strip()
            
            predictions.append(prediction)
            
            if len(predictions) % 10 == 0:
                logging.info(f"Processed {len(predictions)}/{len(test_data)} samples")
        
        # Evaluate
        result = self.evaluate_predictions(true_labels, predictions,
                                          "Gemini-2.0-Flash", "chain-of-thought")
        self.results['gemini_cot'] = result
        
        return result
    
    def run_all_experiments(self, sample_size: int = None):
        """Run all Gemini experiments"""
        print("\n" + "="*60)
        print("GEMINI 2.0 FLASH EXPERIMENTS")
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
        self.save_results("gemini_results")
        
        return self.results


def main():
    """Main execution"""
    import sys
    
    print("="*60)
    print("GEMINI 2.0 FLASH - LAMUS EXPERIMENTS")
    print("="*60)
    
    # Get API key
    api_key = input("Enter your Gemini API key: ").strip()
    
    if not api_key:
        print("API key required!")
        sys.exit(1)
    
    # Initialize classifier
    classifier = GeminiClassifier(
        train_path="train_final.csv",
        test_path="test_final.csv",
        api_key=api_key
    )
    
    # Ask for sample size (for testing)
    sample_input = input("\nEnter sample size for testing (or press Enter for full dataset): ").strip()
    sample_size = int(sample_input) if sample_input else None
    
    if sample_size:
        print(f"\n⚠️  Running experiments on {sample_size} samples (testing mode)")
    else:
        print(f"\n📊 Running experiments on full test set ({len(classifier.test_df)} samples)")
        confirm = input("This will take several hours and use API credits. Continue? (y/n): ")
        if confirm.lower() != 'y':
            print("Exiting...")
            sys.exit(0)
    
    # Run all experiments
    results = classifier.run_all_experiments(sample_size)
    
    print("\n✅ All Gemini experiments complete!")
    print(f"Results saved to gemini_results/")


if __name__ == "__main__":
    main()
