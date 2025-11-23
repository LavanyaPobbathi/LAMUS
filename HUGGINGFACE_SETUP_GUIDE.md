# HuggingFace API Setup Guide for LAMUS

## 🔑 Step-by-Step Setup

### 1. Create HuggingFace Account
1. Go to https://huggingface.co/join
2. Sign up with your email
3. Verify your email address

### 2. Get Your API Token
1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name it "LAMUS-Research"
4. Select "Read" permission (minimum required)
5. Copy the token (starts with `hf_...`)
6. **Save it securely - you won't see it again!**

### 3. Request Model Access

#### 🔓 Open Access Models (Work Immediately)
- ✅ **SaulLM-7B**: https://huggingface.co/Equall/Saul-7B-Instruct-v1
- ✅ **Law-LLM**: https://huggingface.co/AdaptLLM/law-LLM
- ✅ **Qwen2.5-7B**: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

#### 🔒 Gated Models (Need Access Request)

**Llama-3-8B** (Meta's model):
1. Go to https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
2. Click "Agree and access repository"
3. Fill out Meta's form (use your .edu email if available)
4. Wait for approval (usually 1-24 hours)
5. Alternative: Try https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

**SaulLM-54B** (Large legal model):
1. Go to https://huggingface.co/Equall/SaulLM-54B-Instruct
2. Request access if required
3. This model may have limited API availability

### 4. Model Availability Status

| Model | Access | API Status | Priority |
|-------|--------|------------|----------|
| Gemini 2.0 Flash | API Key | ✅ Ready | 1st |
| SaulLM-7B | Open | ✅ Ready | 2nd |
| Law-LLM-7B | Open | ✅ Ready | 3rd |
| Qwen2.5-7B | Open | ✅ Ready | 4th |
| Llama-3.1-8B | Gated | ⏳ Request | 5th |
| SaulLM-54B | Gated | ⏳ Request | 6th |

## 🎯 Recommended Approach for Your Paper

### Phase 1: Immediate Testing (No Access Needed)
1. **Gemini 2.0 Flash** - Use Google API
2. **SaulLM-7B** - Open access on HF
3. **Law-LLM** - Open access on HF
4. **Qwen2.5-7B** - Open access on HF

### Phase 2: After Access Approval
5. **Llama-3.1-8B** - After Meta approval
6. **SaulLM-54B** - If accessible

## 📊 Running Experiments

### Quick Start (Test with 50 samples)
```bash
# Test with SaulLM-7B (works immediately)
python model_huggingface_api.py
> Choice: 1 (Test single model)
> Select model: 1 (SaulLM-7B)
> Sample size: 50
```

### Full Run (After testing)
```bash
# Run all accessible models
python model_huggingface_api.py
> Choice: 3 (Legal domain models)
> Sample size: [Enter for full]
```

## 🚨 Priority Actions

1. **NOW**: Get HF token from https://huggingface.co/settings/tokens
2. **NOW**: Start Gemini experiments (separate script)
3. **NOW**: Run SaulLM-7B (open access)
4. **TODAY**: Request Llama-3 access for tomorrow
5. **TONIGHT**: Run experiments overnight on full dataset
