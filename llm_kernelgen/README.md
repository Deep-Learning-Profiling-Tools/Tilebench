# TileBench-LLMGen

An LLM kernel generation and evaluation extension to TileBench.

Evaluates model-generated Triton and cuTile kernels under **controlled prompt
visibility** and **identical correctness/performance criteria** as hand-written
baselines.

---

## Overview

```
LLM receives:                         LLM never sees:
  ✓ impl_torch.py                       ✗ impl_triton.py
  ✓ config.yaml                         ✗ impl_cutile.py
  ✓ input generator function            ✗ benchmark results
  ✓ problem statement (if exists)       ✗ historical PR notes / fixes
  ✓ DSL reference (curated cookbook)    ✗ Nsight profiles
  ✓ few-shot examples (train split)
```

---

## Installation

```bash
pip install openai jinja2 pyyaml
```

Required environment variables:

```bash
# OpenAI official (Responses API)
export OPENAI_API_KEY="sk-..."

# Any OpenAI-compatible endpoint (optional)
export OPENAI_COMPAT_API_KEY="..."
export OPENAI_COMPAT_BASE_URL="https://your-endpoint/v1"
```

---

## Step-by-step usage

### 1. Configure a model

Edit [`configs/models.yaml`](configs/models.yaml).  Built-in aliases include
`gpt4o`, `o3_medium`, `o3_high`, `o4_mini_medium`, and `compat_coder`.

To add your own:

```yaml
models:
  my_model:
    provider: openai          # or openai_compat / local_vllm
    model: "gpt-4o-mini"
    temperature: 0.2
    max_output_tokens: 8192
```

### 2. Define an experiment

Edit [`configs/experiments.yaml`](configs/experiments.yaml):

```yaml
experiments:
  my_exp:
    model_alias: my_model
    backends: [triton]
    prompt_profile: triton_zero_shot
    operators_split: dev_ops   # train_examples | dev_ops | test_ops
    num_samples: 3
    repair_rounds: 2
    description: "My first experiment"
```

### 3. Run the experiment

```bash
PYTHONPATH=. python scripts/eval_llm_kernels.py --experiment my_exp
```

This performs: **generate → evaluate → repair → summarize**.

### 4. Inspect results

```
llm_kernelgen/generated/my_exp/
├── summary.json                    # Aggregated metrics
├── batch_results.json              # Per-sample success/fail list
└── softmax/triton/sample_00/
    ├── prompt.md                   # Exact prompt sent to LLM
    ├── response.raw.txt            # Raw LLM response
    ├── impl_triton.py              # Extracted kernel code
    ├── metadata.json               # Model, tokens, latency, context audit
    ├── bench.json                  # Per-case timing results
    ├── eval_status.json            # Stage reached + error details
    └── eval.log                    # Subprocess stdout/stderr
```

---

## Individual commands

### Generate only

```bash
PYTHONPATH=. python llm_kernelgen/scripts/generate.py \
    --operator softmax rmsnorm \
    --backend triton \
    --model o3_medium \
    --experiment exp_test \
    --samples 5 \
    --profile triton_few_shot_4
```

### Evaluate one generated file

```bash
PYTHONPATH=. python scripts/run_generated.py \
    --operator softmax \
    --backend triton \
    --impl llm_kernelgen/generated/exp_test/softmax/triton/sample_00/impl_triton.py
```

### Repair a failed sample

```bash
PYTHONPATH=. python llm_kernelgen/scripts/repair.py \
    --operator softmax \
    --backend triton \
    --sample-dir llm_kernelgen/generated/exp_test/softmax/triton/sample_00 \
    --model o3_medium \
    --max-rounds 2
```

### Summarize an experiment

```bash
PYTHONPATH=. python llm_kernelgen/scripts/summarize.py \
    --experiment exp_test
```

---

## Dataset splits

| Split | Purpose | LLM access |
|---|---|---|
| `train_examples` | Few-shot demonstrations | ✅ code shown in prompt |
| `dev_ops` | Prompt engineering / smoke tests | ❌ code never shown |
| `test_ops` | Final paper evaluation | ❌ code never shown |

Edit [`dataset/manifest.yaml`](dataset/manifest.yaml) to update the splits.

---

## Evaluation stages

| # | Stage name | Passes when |
|---|---|---|
| 0 | `syntax_import` | File parses and imports without error |
| 1 | `run_case0` | `run()` executes on case 0 without exception |
| 2 | `correct_case0` | Output matches `impl_torch` on case 0 |
| 3 | `correct_all` | Correctness passes on all benchmark cases |
| 4 | `bench_complete` | Timing suite completes successfully |
| 5 | `perf_score` | Performance metrics computed |

---

## Metrics reported

**Generation quality**

| Metric | Definition |
|---|---|
| `syntax_pass_rate` | Fraction of samples that pass syntax/import |
| `runtime_pass_rate` | Fraction where `run()` executes without exception |
| `correctness_pass_rate` | Fraction that match `impl_torch` on all cases |
| `pass@1` | Probability that a single sample is correct |
| `pass@k` | Unbiased estimator (Chen et al. 2021) |
| `repair_success_rate` | Fraction of failed samples fixed within R rounds |

**Performance** (correctness-passing only)

| Metric | Definition |
|---|---|
| `mean_speedup_vs_torch` | LLM kernel vs PyTorch baseline |
| `median_speedup_vs_torch` | Median across correct samples |

**Effort / cost**

| Metric | Definition |
|---|---|
| `mean_prompt_tokens` | Average input tokens per sample |
| `mean_completion_tokens` | Average output tokens per sample |
| `mean_reasoning_tokens` | Average reasoning tokens (o-series) |
| `mean_latency_s` | Average API round-trip time |
| `mean_loc` | Average lines of code in generated file |

---

## Prompt profiles

Defined in [`configs/prompt_profiles.yaml`](configs/prompt_profiles.yaml):

| Profile | Examples | Description |
|---|---|---|
| `triton_zero_shot` | 0 | DSL reference only |
| `triton_few_shot_2` | 2 | 2 train-split examples |
| `triton_few_shot_4` | 4 | 4 train-split examples |
| `cutile_zero_shot` | 0 | DSL reference only |
| `cutile_few_shot_2` | 2 | 2 train-split examples |
| `cutile_few_shot_4` | 4 | 4 train-split examples |

---

## Adding a new model provider

1. Add a provider entry to `configs/models.yaml`:
   ```yaml
   providers:
     my_provider:
       api_type: chat_completions
       base_url: "https://api.my-provider.com/v1"
       api_key_env: "MY_PROVIDER_KEY"
   ```
2. Add a model alias pointing to that provider.
3. For reasoning models on non-OpenAI endpoints, use `extra_body`:
   ```yaml
   models:
     my_reasoning_model:
       provider: my_provider
       model: "model-id"
       temperature: 0.2
       max_output_tokens: 16000
       extra_body:
         thinking:
           type: "enabled"
           budget_tokens: 8000
   ```

---

## Reproducibility checklist

Before publishing results, verify:

- [ ] `metadata.json` for every sample contains `"forbidden_context_enforced": true`
- [ ] No `impl_triton.py` / `impl_cutile.py` paths appear in any `prompt.md`
- [ ] `git_commit` field in `metadata.json` matches the paper's code version
- [ ] Test operator split (`test_ops.yaml`) was fixed before any generation
- [ ] Few-shot examples are exclusively from `train_examples` split
- [ ] Repair prompts contain only the previously generated code + error log
- [ ] `generated/` directory is **not** committed to the repository
