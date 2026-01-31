#!/usr/bin/env python3
"""
LAMUS - Label ALL Courts (Sharded, 4 GPUs)
==========================================
Labels all courts EXCEPT Roberts Court (already done).

Run 4 processes (one per GPU):

CUDA_VISIBLE_DEVICES=0 nohup python3 E_label_all_courts_sharded.py 0 4 > E_shard0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python3 E_label_all_courts_sharded.py 1 4 > E_shard1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python3 E_label_all_courts_sharded.py 2 4 > E_shard2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python3 E_label_all_courts_sharded.py 3 4 > E_shard3.log 2>&1 &

Monitor:
tail -f E_shard0.log E_shard1.log E_shard2.log E_shard3.log

After all complete, merge:
python3 E_merge_all_courts.py
"""

import os
import sys
import json
import time
import gc
from datetime import datetime

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")
os.environ.setdefault("HF_HOME", "/home/lavanya/.cache/huggingface")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================
HF_TOKEN = "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"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

BEST_MODEL_PATH = "./best_model_85"
BACKUP_MODEL_PATH = "./lamus_finetuned_final"

# INPUT: All courts except Roberts (created by filter script)
SCOTUS_INPUT = "scotus_labeled/all_courts_sentences.csv"
OUTPUT_DIR = "scotus_labeled"

SAVE_EVERY = 10000  # Save checkpoint every 10k
EARLY_FLUSH = 100
MAX_LENGTH = 128
MAX_NEW_TOKENS = 10

LABELS = ['Facts', 'Issue', 'Rule/Law/Holding', 'Analysis', 'Conclusion', 'Others']
ASSISTANT_PREFIX = "<|start_header_id|>assistant<|end_header_id|>\n\n"
# =========================


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_gpu_memory():
    if not torch.cuda.is_available():
        return "N/A"
    allocated = torch.cuda.memory_allocated(0) / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"{allocated:.1f}/{total:.1f}GB"


def format_prompt(sentence: str) -> str:
    sentence = (sentence or "").strip()
    if len(sentence) > 200:
        sentence = sentence[:200] + "..."
    return f"Classify: Facts/Issue/Rule/Analysis/Conclusion/Others\n\n{sentence}\n\nCategory:"


def resolve_model_path():
    cand = [
        (BEST_MODEL_PATH, "85.16%"),
        (BACKUP_MODEL_PATH, "80.37%"),
    ]
    for path, acc in cand:
        if os.path.exists(path) and os.path.exists(os.path.join(path, "adapter_model.safetensors")):
            return path, acc
    return None, None


def load_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    print("\n" + "=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    model_path, model_acc = resolve_model_path()
    if model_path is None:
        print("❌ No adapter found.")
        sys.exit(1)

    print(f"✅ Using adapter: {model_path} (expected {model_acc})")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    print(f"✅ Model loaded. GPU mem: {get_gpu_memory()}")
    return model, tokenizer, model_path, model_acc


def classify_sentence(model, tokenizer, sentence: str) -> str:
    prompt = format_prompt(sentence)
    input_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>{ASSISTANT_PREFIX}"

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    for label in LABELS:
        if label.lower() in response.lower():
            return label
    return "Others"


def load_checkpoint(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"done_ids": []}


def save_checkpoint(path, done_ids):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"done_ids": list(done_ids), "timestamp": datetime.now().isoformat()}, f)
    os.replace(tmp, path)


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 E_label_all_courts_sharded.py <shard_id> <num_shards>")
        sys.exit(1)

    shard_id = int(sys.argv[1])
    num_shards = int(sys.argv[2])

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    shard_out = os.path.join(OUTPUT_DIR, f"all_courts_labeled_shard{shard_id}.csv")
    shard_partial = shard_out + ".partial"
    shard_ckpt = os.path.join(OUTPUT_DIR, f"all_courts_checkpoint_shard{shard_id}.json")

    print("=" * 70)
    print("LAMUS - ALL COURTS LABELING (SHARDED)")
    print(f"Started: {datetime.now()}")
    print(f"Shard: {shard_id}/{num_shards}")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        sys.exit(1)

    print(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")

    if not os.path.exists(SCOTUS_INPUT):
        print(f"❌ Input not found: {SCOTUS_INPUT}")
        print("   Run the filter script first!")
        sys.exit(1)

    print(f"\n📥 Loading: {SCOTUS_INPUT}")
    df = pd.read_csv(SCOTUS_INPUT, low_memory=False)
    print(f"   Total rows: {len(df):,}")

    # Get shard
    shard_df = df.iloc[shard_id::num_shards].copy()
    shard_df.reset_index(drop=True, inplace=True)
    print(f"   Shard {shard_id} rows: {len(shard_df):,}")

    # Courts in this shard
    print(f"\n📊 Courts in shard:")
    print(shard_df['court'].value_counts())

    # Load model
    clear_memory()
    model, tokenizer, model_path, model_acc = load_model()

    # Resume support
    done_ids = set()
    if os.path.exists(shard_ckpt):
        ck = load_checkpoint(shard_ckpt)
        done_ids = set(ck.get("done_ids", []))
        print(f"\n📂 Resuming: {len(done_ids):,} already done")

    if os.path.exists(shard_partial):
        try:
            part = pd.read_csv(shard_partial)
            if "row_id" in part.columns:
                done_ids = set(part["row_id"].tolist())
                print(f"📄 Partial CSV: {len(done_ids):,} rows")
        except:
            pass

    output_rows = []
    start_time = time.time()
    labeled_count = 0

    pbar = tqdm(total=len(shard_df), desc=f"Shard {shard_id}", dynamic_ncols=True)
    
    try:
        for i, row in shard_df.iterrows():
            rid = int(row["row_id"])
            if rid in done_ids:
                pbar.update(1)
                continue

            sent = str(row["sentence"]) if pd.notna(row["sentence"]) else ""
            if not sent.strip():
                pred = "Others"
            else:
                pred = classify_sentence(model, tokenizer, sent)

            out = row.to_dict()
            out["Predicted_Label"] = pred
            output_rows.append(out)
            done_ids.add(rid)
            labeled_count += 1

            # Early flush
            if labeled_count == EARLY_FLUSH:
                pd.DataFrame(output_rows).to_csv(shard_partial, index=False)
                save_checkpoint(shard_ckpt, done_ids)
                print(f"\n✅ Early flush: {labeled_count} rows")

            # Periodic save
            if labeled_count % SAVE_EVERY == 0:
                pd.DataFrame(output_rows).to_csv(shard_partial, index=False)
                save_checkpoint(shard_ckpt, done_ids)

                elapsed = time.time() - start_time
                rate = labeled_count / elapsed if elapsed > 0 else 0
                remaining = len(shard_df) - len(done_ids)
                eta_h = (remaining / rate / 3600) if rate > 0 else 0

                print(f"\n📊 Shard {shard_id}: {len(done_ids):,}/{len(shard_df):,} ({100*len(done_ids)/len(shard_df):.1f}%)")
                print(f"   Rate: {rate:.1f}/sec | ETA: {eta_h:.1f}h | GPU: {get_gpu_memory()}")

            pbar.update(1)

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted - saving...")
        if output_rows:
            pd.DataFrame(output_rows).to_csv(shard_partial, index=False)
        save_checkpoint(shard_ckpt, done_ids)
        sys.exit(0)

    finally:
        pbar.close()

    # Final save
    final_df = pd.DataFrame(output_rows)
    final_df.to_csv(shard_out, index=False)

    if os.path.exists(shard_partial):
        os.remove(shard_partial)
    if os.path.exists(shard_ckpt):
        os.remove(shard_ckpt)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"✅ SHARD {shard_id} COMPLETE!")
    print("=" * 70)
    print(f"Rows: {len(final_df):,}")
    print(f"Output: {shard_out}")
    print(f"Time: {elapsed/3600:.2f} hours")


if __name__ == "__main__":
    main()