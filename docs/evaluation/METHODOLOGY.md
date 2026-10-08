# LDMARK Evaluation Methodology

**Version:** 1.0  
**Status:** Experimental  
**Last Updated:** 2024

---

## Overview

This document explains the LDMARK evaluation methodology for measuring how model compression affects language model behavior.

**Critical Principle:** This framework measures **specific, reproducible behavioral differences** on a **specific model and evaluation suite**. It does NOT produce universal quality scores, does NOT claim to predict large-model behavior, and does NOT compare against other compression methods using different models or settings.

---

## What Each Metric Measures

### 1. Token Match Rate
**Measures:** Percentage of generated tokens that exactly match between baseline and compressed model outputs.

**What it tells you:** How often the compressed model produces the same token as the baseline at each position.

**What it does NOT tell you:** Semantic equivalence. Different tokens can produce the same meaning.

### 2. Exact Output Match Rate
**Measures:** Percentage of prompts where the complete generated output exactly matches (character-for-character).

**What it tells you:** How often compression produces identical output sequences.

**What it does NOT tell you:** Near-miss quality. A single token difference (e.g., "Paris" vs "paris") counts as complete failure.

### 3. Logit Mean Absolute Error (MAE)
**Measures:** Average absolute difference between logit values at each position: `mean(|logits_baseline - logits_compressed|)`.

**What it tells you:** Average numerical deviation in the model's raw predictions.

**What it does NOT tell you:** Whether the deviation changes the generated token. Small logit changes may not affect sampling.

### 4. Logit Cosine Similarity
**Measures:** Cosine similarity between baseline and compressed logit vectors at each position.

**What it tells you:** Directional similarity of the probability distributions, independent of magnitude.

**What it does NOT tell you:** Absolute calibration. Two distributions can be directionally similar but have different sharpness.

### 5. Top-k Agreement
**Measures:** Percentage of positions where the top-k predicted tokens overlap between baseline and compressed.

**What it tells you:** Whether the models agree on the most likely continuations.

**What it does NOT tell you:** Probability mass differences within the top-k.

### 6. Perplexity Ratio
**Measures:** Ratio of compressed model perplexity to baseline perplexity: `PPL_compressed / PPL_baseline`.

**What it tells you:** Relative increase in uncertainty/loss due to compression.

**What it does NOT tell you:** Whether the increased perplexity corresponds to user-noticeable quality degradation.

---

## What These Metrics Do NOT Measure

| Metric | Does NOT Measure |
|--------|------------------|
| Token Match Rate | Semantic equivalence, factual correctness, user satisfaction |
| Exact Output Match | Near-miss quality, minor variations (case, punctuation) |
| Logit MAE | Downstream task performance, human preference |
| Logit Cosine | Absolute probability calibration |
| Top-k Agreement | Probability mass distribution within top-k |
| Perplexity Ratio | Task-specific performance, generation quality |

---

## Why Tensor Error ≠ Behavioral Quality

This is a **critical distinction** that this framework makes explicit:

### Tensor Reconstruction Error
- Measures: Numerical difference between original and reconstructed weight tensors
- Domain: Weight space (parameters)
- Typical metrics: MAE, MSE, cosine similarity, relative error

### Model Behavioral Change
- Measures: Difference in model output behavior (tokens, logits, perplexity)
- Domain: Output space (generations, predictions)
- Typical metrics: Token match, exact match, logit MAE, top-k agreement, perplexity ratio

### Why They Are Different

1. **Non-linear amplification:** Small weight errors can be amplified through deep network layers
2. **Error cancellation:** Multiple weight errors can cancel out in the forward pass
3. **Task-specific sensitivity:** Some tasks are robust to weight noise; others are extremely sensitive
4. **Compensatory effects:** Attention mechanisms and residual connections can compensate for weight perturbations

### What This Means

- **Low tensor error ≠ Identical behavior:** A model can have excellent weight reconstruction but different outputs
- **High tensor error ≠ Catastrophic failure:** A model can have poor weight reconstruction but similar outputs
- **No universal correlation:** The relationship depends on model architecture, task, and compression method

**This framework measures BOTH separately and does not infer one from the other.**

---

## Limitations of Small-Model Experiments

### 1. Scale Effects
- Small models (e.g., 1M-100M parameters) have different sensitivity to compression than large models (7B+)
- Attention patterns, feature hierarchy, and redundancy differ significantly with scale
- **Do not extrapolate** small-model results to large models

### 2. Synthetic Backend
- Current implementation uses `SyntheticModelBackend` for testing without real models
- Tokenization is character-based, not BPE/SentencePiece
- Logits are synthetically generated, not from actual forward passes
- **Results are NOT representative of real model behavior**

### 3. Prompt Suite Size
- 25 prompts across 5 categories
- Not comprehensive; misses many behavioral dimensions
- No open-ended generation, multi-turn dialogue, or complex reasoning

### 4. Deterministic Generation Only
- Uses temperature=0.0, top-k=1 for reproducibility
- Does not evaluate sampling behavior, diversity, or temperature sensitivity

### 4. No Downstream Tasks
- No evaluation on benchmarks (MMLU, GSM8K, HumanEval, etc.)
- No evaluation on real-world use cases (summarization, translation, etc.)

---

## Why Benchmark Results Should Not Be Generalized

### Model-Specific
- Results are for **one specific model** (synthetic test model in current implementation)
- Different architectures (LLaMA, GPT, BERT, etc.) respond differently to compression
- Training data, tokenizer, and hyperparameters all affect compression sensitivity

### Compression-Specific
- Results are for **specific compression configurations** (group size, scale dtype, symmetric, etc.)
- INT8 at group_size=128 ≠ INT8 at group_size=32
- Binary/ternary are research formats with no production kernels

### Evaluation-Specific
- Results are for **this specific prompt suite and settings**
- Different prompts, temperatures, max_tokens, or sampling strategies yield different results
- Token match rate at temp=0.0 ≠ token match rate at temp=0.7

### Hardware/Software Specific
- Numerical differences can arise from different BLAS libraries, CUDA versions, etc.
- Quantization calibration data affects results

---

## Proper Interpretation Framework

### Correct Statement
> "At INT4 compression (group_size=128, symmetric, FP16 scales) on the synthetic test model with the LDMARK prompt suite (25 prompts, temp=0.0), we observed: 92.3% token match rate, 68% exact match rate, 0.023 mean logit MAE, 0.987 mean logit cosine similarity, 0.91 top-1 agreement, and 1.04x perplexity ratio."

### Incorrect Statements
> ❌ "INT4 compression preserves 92% of model quality."
> ❌ "This compression method is 95% as good as FP16."
> ❌ "Binary compression reduces intelligence by 15%."
> ❌ "These results predict 27B model behavior."

---

## Reproducibility Requirements

Every evaluation must record:

1. **Seed** - Random seed for all stochastic operations
2. **Software Versions** - LDMARK, NumPy, PyTorch, Transformers, Python
3. **Model Identifier** - Exact model name, source, and version
4. **Compression Configuration** - Method, group size, scale dtype, symmetric, zero point
5. **Evaluation Configuration** - Prompts, task types, generation settings
6. **Timestamp** - ISO 8601 format
7. **Hardware** - CPU, GPU, RAM, OS

Where deterministic behavior is impossible (e.g., GPU non-determinism, CUDA kernel variations), this MUST be explicitly stated in the report limitations.

---

## Evaluation Pipeline

```
Model + Artifact
       │
       ▼
Runtime (loads compressed weights)
       │
       ▼
Baseline Generation (FP16/original)
       │
       ├─── Prompt 1 ───▶ Output 1, Tokens 1, Logits 1
       ├─── Prompt 2 ───▶ Output 2, Tokens 2, Logits 2
       │
       ▼
Compressed Generation (INT8/INT4/etc.)
       │
       ├─── Prompt 1 ───▶ Output 1', Tokens 1', Logits 1'
       ├─── Prompt 2 ───▶ Output 2', Tokens 2', Logits 2'
       │
       ▼
Comparison
       │
       ├─── Token Match Rate
       ├─── Exact Output Match
       ├─── Logit MAE / Cosine / Top-k
       ├─── Perplexity Ratio
       │
       ▼
Structured Report (JSON + Markdown)
```

---

## Future Extensions

This methodology is designed to be extended:

1. **Real Model Backends** - Integration with HuggingFace, llama.cpp, vLLM
2. **Larger Prompt Suites** - MMLU, GSM8K, HumanEval, MT-Bench
3. **Sampling Evaluation** - Temperature sweeps, diversity metrics
4. **Downstream Tasks** - Fine-tuning after compression, few-shot evaluation
5. **Human Evaluation** - Side-by-side comparison, preference testing
6. **Long-form Generation** - Multi-turn, coherence, consistency

Each extension must maintain the core principles: **no universal quality claims, explicit limitations, reproducible measurements.**

---

## References

- LDMARK Artifact Format Specification
- LDMARK Runtime Contract
- Compression Metrics (src/ldmark/compression/metrics.py)
- Prompt Suite (src/ldmark/evaluation/prompts.py)

---

**Remember:** The goal is not to produce a single number that "proves" compression quality. The goal is to provide transparent, reproducible measurements that let researchers and engineers make informed decisions about compression trade-offs for their specific use case.