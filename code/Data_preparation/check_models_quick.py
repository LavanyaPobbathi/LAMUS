# Save as check_models_quick.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

HF_TOKEN = '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'

models_to_check = [
    ("Llama-3-8B", "meta-llama/Meta-Llama-3-8B-Instruct"),
    ("Llama-2-7B", "meta-llama/Llama-2-7b-chat-hf"),
    ("Phi-3-mini", "microsoft/Phi-3-mini-4k-instruct"),
    ("Mistral-7B", "mistralai/Mistral-7B-Instruct-v0.2"),
    ("SaulLM-7B", "Equall/Saul-7B-Instruct-v1"),
    ("Law-LLM", "AdaptLLM/law-LLM"),
    ("Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct"),
]

print("Checking which models are accessible...")
print("="*50)

working_models = []

for name, path in models_to_check:
    try:
        print(f"Checking {name}...", end=" ")
        # Just try loading tokenizer (quick test)
        tokenizer = AutoTokenizer.from_pretrained(path, token=HF_TOKEN)
        print("✅ WORKS")
        working_models.append((name, path))
    except Exception as e:
        if "404" in str(e):
            print("❌ NOT FOUND")
        elif "401" in str(e) or "403" in str(e):
            print("🔒 NO ACCESS")
        else:
            print(f"❌ ERROR: {str(e)[:30]}")

print("\n" + "="*50)
print(f"Working models: {len(working_models)}/{len(models_to_check)}")
for name, path in working_models:
    print(f"  ✅ {name}")