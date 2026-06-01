# NVIDIA Nemotron Reasoning Challenge — Implementation Plan

## Overview

Fine-tune **Nemotron-3-Nano-30B** with a LoRA adapter (rank ≤ 32) to solve logical reasoning puzzles
(bit manipulation, algebra, word transformations, Roman numerals, etc.).

The benchmark metric is **Accuracy** — the fraction of `\boxed{}` answers that match ground truth
exactly or within a small numerical tolerance.

**Execution environment:** Local workstation with RTX 6000 (Ada Generation, ~48GB VRAM) (No Internet)
**Training library:** Native PyTorch with HuggingFace PEFT and Transformers Trainer (Unsloth incompatible with Mamba, and no trl available offline)
**CoT generation:** Gemini free API (gemini-2.0-flash or gemini-2.5-flash) — *or* alternative
(Groq / OpenAI) if Gemini quota is exhausted

---

## Phase 0 EDA Findings (Completed ✅)

> [!IMPORTANT]
> These findings should drive all subsequent decisions. Read before writing any code.

### Dataset
| Stat | Value |
|---|---|
| Training examples | **9,500** |
| Sample test examples | **3** (real test = "several hundred" per rules) |
| Exact duplicates | 0 |
| Contradictions | 0 |

### Puzzle Category Breakdown
| Category | Count | % | Avg Prompt Tokens | Avg Answer Length |
|---|---|---|---|---|
| `decimal_math` | 3,191 | 33.6% | ~64 tokens | 5.0 chars |
| `binary` | 1,993 | 21.0% | ~90 tokens | 11.4 chars |
| `roman_numerals` | 1,582 | 16.7% | ~52 tokens | 4.1 chars |
| `word_sequence` | 1,180 | 12.4% | ~78 tokens | **25.4 chars** |
| `other` | 868 | 9.1% | ~48 tokens | 2.9 chars |
| `integer_math` | 686 | 7.2% | ~48 tokens | 3.0 chars |

### Key Structural Insights
1. **All prompts** start with `"In Alice's Wonderland, "` — single themed benchmark
2. **Prompts are SHORT** — max 510 chars, p95 only ~96 tokens. 4096 max_seq_length gives massive room for CoT
3. **Each prompt already contains 5–10 few-shot examples** in `input -> output` format, then asks `"Now, determine the output for: X"`
4. **The model's job is rule induction**: given examples, find the hidden rule, apply to new input
5. **83.4% single-word answers** — binary strings, numbers, roman numerals (easy to match exactly)
6. **16.6% multi-word answers** — word decryption sequences (harder, `\boxed{}` must capture full phrase)

### Strategic Implications for Training
- **Binary puzzles** (21%): Rule varies per puzzle (XOR, shift, rotation, majority). CoT must identify the specific rule from examples. Hardest category.
- **Decimal math** (33.6%): Mostly unit conversions and physics formulas. Pattern is consistent — multiply by some constant. CoT: find the constant.
- **Roman numerals** (16.7%): Pure lookup. Almost deterministic — model probably already knows this.
- **Word sequences** (12.4%): Caesar cipher or substitution cipher. CoT: decode the mapping.
- **Integer math** (7.2%): Custom operator rules (e.g., `64-65 = 201` means reverse and concatenate digits).
- **Other** (9.1%): Symbol manipulation — likely hardest after binary.

### CoT Generation Priority
Generate CoT in this order (hardest → most value):
1. `binary` — highest variance rules, hardest to reason about
2. `other` — symbol manipulation, opaque rules
3. `word_sequence` — multi-word output, cipher decoding
4. `integer_math` — custom operator rules
5. `decimal_math` — find constant multiplier (simple)
6. `roman_numerals` — model likely already knows this; deprioritize

---

## Strategy: 4-Phase Escalation

```
Phase 0 (EDA)  →  Phase 1 (Baseline SFT)  →  Phase 2 (CoT SFT)  →  Phase 3 (GRPO)
Understand data    Quick leaderboard score     Big accuracy jump       RL push (bonus)
```

Each phase produces a real submission. We always have something to submit.

---

## Phase 0 — Exploratory Data Analysis

**Goal:** Understand puzzle types and volume *before* touching any model. Done in a lightweight notebook — no GPU needed.

**Notebook:** `phase0_eda.ipynb`

**Steps:**
1. Load `train.csv` with pandas
2. Count total puzzles; inspect `prompt` and `answer` columns
3. Categorize puzzle types (heuristic keyword matching):
   - Bit manipulation (binary strings, XOR, shifts)
   - Algebraic equations
   - Roman numerals
   - Word / sequence transformations
   - Other
4. Plot category distribution
5. Measure answer length distribution (digits, binary, strings)
6. Identify any duplicates or near-duplicates
7. Sample 5–10 examples from each category for manual inspection
8. Check whether answers are always numeric vs. mixed-type

**Key questions to answer before Phase 1:**
- How many training examples do we have?
- Are all puzzle types amenable to the same prompt format?
- Is the answer always a single token or multi-token?

---

## Phase 1 — Prompting Baseline (Raw SFT, No CoT)

**Goal:** Get a real leaderboard score as fast as possible. Sets a floor we can beat.

**Notebook:** `phase1_baseline_sft.ipynb`

### What to do
1. Load Nemotron-3-Nano-30B with PyTorch and Transformers (bfloat16, device_map="auto")
2. Initialize a LoRA adapter (rank = 32, max allowed)
3. Train on raw `(prompt, answer)` pairs from `train.csv` — no reasoning chains
4. Save adapter → zip → submit

### Training input format (per example)
```
<system>
You are an expert logical reasoning assistant.
Think step by step.
Always place your final answer inside \boxed{}.
</system>
<user>
{prompt}
</user>
<assistant>
\boxed{{answer}}
</assistant>
```

> [!NOTE]
> Even without CoT, fine-tuning on (prompt, answer) teaches the model to output answers in
> `\boxed{}` format — which is required by the scoring metric.

### Training config (RTX 6000 Ada)

| Parameter | Recommended Value |
|---|---|
| `load_in_4bit` | False (bfloat16 native + CPU offloading) |
| `lora_r` | 32 |
| `lora_alpha` | 64 |
| `lora_dropout` | 0.05 |
| `target_modules` | in_proj, out_proj, up_proj, down_proj |
| `per_device_train_batch_size` | 4 |
| `gradient_accumulation_steps` | 4 (effective batch = 16) |
| `learning_rate` | 2e-4 |
| `max_seq_length` | 2048 |
| `num_train_epochs` | 3 |
| `lr_scheduler_type` | cosine |

**Expected result:** Low-moderate score. Gives us a real baseline to beat in Phase 2.

---

## Phase 2 — SFT with Chain-of-Thought Reasoning

**Goal:** Teach the model to reason step-by-step. Expected to be the biggest accuracy jump.
**Baseline to beat:** LB **0.63**, local **0.561**

> [!IMPORTANT]
> Phase 1 used a custom `<|system|>...<|user|>...<|assistant|>` format. The **metric evaluator
> uses `apply_chat_template(..., enable_thinking=True)`** which produces a different prompt structure.
> Phase 2 training must match this format, otherwise we're training in a different distribution
> than what the model is evaluated in.

### 🔑 Key Change from Phase 1: Training Format

**Phase 1 format (wrong for evaluation):**
```
<|system|>
{system_prompt}
<|user|>
{puzzle_prompt}
<|assistant|>
\boxed{answer}
```

**Phase 2 format (matches evaluation):**
Use the tokenizer's own `apply_chat_template` with `enable_thinking=True`:
```python
# Build the training string using the tokenizer's chat template
user_content = puzzle_prompt  # evaluator appends \boxed instruction itself
assistant_content = f"<think>\n{cot_reasoning}\n</think>\n\\boxed{{{answer}}}"

# Apply the same template the evaluator uses
full_text = tokenizer.apply_chat_template(
    [
        {'role': 'user', 'content': user_content},
        {'role': 'assistant', 'content': assistant_content},
    ],
    tokenize=False,
    enable_thinking=True,
)
```

**Why this matters:**
- Nemotron-H has a dedicated `<think>` token / thinking mode built in
- Training with `<think>...</think>` teaches the model to use its scratchpad before answering
- The 7% LB vs local gap suggests the model already *wants* to think when given room
- `max_tokens=3584` in evaluation is designed for reasoning chains

---

### Step 2a — Proxy Model Validation (Qwen2.5-7B)

Before spending Kaggle GPU hours on the 30B model, validate the pipeline on a smaller model.

**Notebook:** `phase2_proxy_sft.ipynb`

- Use **Qwen2.5-7B-Instruct** (fits easily on RTX 6000)
- Apply `apply_chat_template` with the equivalent thinking format
- Train for 1 epoch, eval on 10% split
- Verify: accuracy > Phase 1 baseline on proxy model → pipeline is correct
- Quick iteration: full Qwen-7B cycle takes ~30 min vs 7h for 30B

> [!TIP]
> Proxy model training takes ~30 min on RTX 6000. Catches format bugs before wasting compute.

---

### Step 2b — CoT Generation (runs on local WSL machine)

> [!IMPORTANT]
> **This notebook does NOT run on Kaggle.** The Kaggle GPU container is offline.
> Run `phase2_cot_generation.ipynb` on your **local WSL machine**, which has internet.
> No GPU needed — just CPU + Gemini API calls. Does not consume Kaggle GPU quota.

**Data flow:**
```
Local WSL (internet, CPU only)         Kaggle (offline GPU container)
──────────────────────────────         ──────────────────────────────
phase2_cot_generation.ipynb            phase2_sft.ipynb
  ↓ Gemini API calls (~9h overnight)     ↓ reads from Dataset
  ↓ saves cot_train.json                 ↓ /kaggle/input/nemotron-cot/
  → upload to Kaggle Dataset  ─────────→
```

**Notebook:** `phase2_cot_generation.ipynb` (local machine, no Kaggle needed)

Use the Gemini free API to generate `<think>...</think>` reasoning chains.
This uses **backsolved rationalization**: give Gemini the puzzle AND the correct answer.

**Prompt template sent to Gemini:**
```
You are solving a logical puzzle step by step.

Puzzle:
{prompt}

The correct answer is: {answer}

Provide a detailed step-by-step reasoning that explains HOW to arrive at this answer.
Your reasoning should:
1. Identify the hidden rule from the given examples
2. Verify the rule holds for ALL examples shown
3. Apply the rule to the test input to get the answer

Format your response as just the reasoning text (no preamble, no final answer line).
The reasoning will be placed inside <think>...</think> tags.
```

**Training record saved per example:**
```json
{
  "id": "00066667",
  "prompt": "...",
  "cot": "Looking at the examples, I notice that each binary string...",
  "answer": "10010111"
}
```

**CoT generation priority (hardest first — most value):**
| Priority | Category | Phase 1 acc | CoT target | Reason |
|---|---|---|---|---|
| 1 | `other` | 3.0% | 50%+ | Completely broken, highest gain |
| 2 | `binary` | 48.6% | 75%+ | Per-puzzle rules, must reason per-example |
| 3 | `integer_math` | 40.0% | 70%+ | Custom operators, needs explicit deduction |
| 4 | `decimal_math` | 52.2% | 85%+ | Find ratio — systematic CoT should dominate |
| 5 | `word_sequence` | 64.7% | 80%+ | Build cipher map, already decent without CoT |
| 6 | `roman_numerals` | **100%** | Skip | Perfect already, don't waste API quota |

**Rate limits & cost:**
- Gemini free tier: ~15 RPM → use `time.sleep(4)` between calls
- ~9500 examples (minus roman_numerals ~1580) = ~7920 to generate
- At 4s/call = ~8.8 hours of generation — run overnight
- Fallback: Groq (llama-3.3-70b-versatile, generous free tier, also free)

> [!IMPORTANT]
> Save the generated CoT dataset to a **Kaggle Dataset** so it persists.
> File name: `cot_train.json`. Never rely on `/kaggle/working/` for persistence.

---

### Step 2c — SFT on Nemotron-30B with CoT

**Notebook:** `phase2_sft.ipynb`

**Training config (same as Phase 1 base, with key changes):**

| Parameter | Phase 1 | Phase 2 Change | Reason |
|---|---|---|---|
| `max_seq_length` | 512 | **2048** | CoT chains are 200–800 tokens |
| `BATCH_SIZE` | 8 | **4** | Longer sequences → more VRAM per example |
| `GRAD_ACCUM` | 2 | **4** | Keep effective batch = 16 |
| `LEARNING_RATE` | 5e-5 | **3e-5** | Slightly lower — fine-tune on top of Phase 1 |
| `NUM_EPOCHS` | 3 | **3** | Same |
| Training format | `\boxed{answer}` | **`<think>cot</think>\n\boxed{answer}`** | Match evaluation |
| Starting checkpoint | Base model | **Phase 1 adapter** | Warm start |

**Warm-starting from Phase 1 adapter:**
```python
# Load base model, then load Phase 1 LoRA weights as initialisation
from peft import PeftModel
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, ...)
model = PeftModel.from_pretrained(model, "phase1_adapter/")  # warm start
model = model.merge_and_unload()  # merge, then apply fresh LoRA for Phase 2
# OR: continue training the same LoRA (simpler)
```

---

### Step 2d — Custom Metric Local Evaluation

> [!IMPORTANT]
> The Kaggle container is **offline** — you cannot `import` from local paths like `utils/`.
> Copy the metric functions **directly into a notebook cell**. No external imports.

**Copy these two functions verbatim into the eval cell of `phase2_sft.ipynb`:**

```python
# ── Copied from utils/nvidia-nemotron-metric.py (competition metric) ──────────
import re, math

def extract_final_answer(text):
    """Competition metric answer extractor. Prioritizes \\boxed{}, then text patterns."""
    if text is None:
        return 'NOT_FOUND'
    boxed_starts = list(re.finditer(r'\\boxed\{', text))
    matches = []
    for i, m in enumerate(boxed_starts):
        start = m.end()
        end = boxed_starts[i + 1].start() if i + 1 < len(boxed_starts) else len(text)
        segment = text[start:end]
        last_brace = segment.rfind('}')
        matches.append(segment[:last_brace] if last_brace != -1 else segment)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        return non_empty[-1] if non_empty else matches[-1].strip()
    for pattern in [
        r'The final answer is:\s*([^\n]+)',
        r'Final answer is:\s*([^\n]+)',
        r'final answer\s*[:：]\s*([^\n]+)',
    ]:
        found = re.findall(pattern, text, re.IGNORECASE)
        if found:
            return found[-1].strip()
    nums = re.findall(r'-?\d+(?:\.\d+)?', text)
    if nums:
        return nums[-1]
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else 'NOT_FOUND'

def verify(stored_answer, predicted):
    """Competition metric grader. Exact string match or 1% numerical tolerance."""
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()
    if re.fullmatch(r'[01]+', stored_answer):   # binary: always exact
        return predicted.lower() == stored_answer.lower()
    try:
        return math.isclose(float(stored_answer), float(predicted),
                            rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored_answer.lower()
# ─────────────────────────────────────────────────────────────────────────────
```

**Use competition inference parameters (not our Phase 1 greedy settings):**

```python
generation_config = dict(
    max_new_tokens=1024,   # allow reasoning chains (not 64)
    do_sample=True,        # stochastic, matches evaluator
    temperature=1.0,       # matches evaluator
    top_p=1.0,             # matches evaluator
)
```

**Expected result:** local eval score should now track LB score within 1–2%.

**Eval speed improvement:** Phase 1 local eval = 200 min on 1900 examples.
- Reduce eval split to **10%** (950 examples) → ~100 min
- Or use batched inference with padded inputs (batch_size=2–4)

---

### Step 2e — Data Augmentation Experiments

Run **two variants** and compare leaderboard scores:

**Variant A — Competition CoT data only**
- Only `cot_train.json` (backsolved CoT from train.csv)
- Clean baseline for measuring CoT benefit

**Variant B — Competition + Public reasoning data**
- 70% competition CoT data
- 30% public reasoning datasets:
  - `AI-MO/NuminaMath-CoT` (math — helps `decimal_math`, `integer_math`)
  - `openai/gsm8k` (grade school math, easy to format as thinking traces)

> [!TIP]
> Submit Variant A first — it's faster and tells us the clean CoT benefit.
> Only run Variant B if time permits after comparing LB scores.

---

## Phase 3 — GRPO (Group Relative Policy Optimization) [BONUS]

**Goal:** Use reinforcement learning to push accuracy beyond SFT. Aspirational — only attempt
if Phase 2 is working well and time remains.

**Notebook:** `phase3_grpo.ipynb`

### Compute Strategy for Phase 3

> [!NOTE]
> Kaggle TPU v3-8 is available for free but we are constrained to offline RTX 6000 hardware.
> Also, **TRL's GRPOTrainer is unavailable offline**. We will need to implement a custom RL loop
> or use purely SFT. **Recommended approach for Phase 3: Stick to supervised learning or implement custom GRPO with pure PyTorch**, since `trl` is unavailable.

### Why GRPO over PPO
- No separate value/critic model → much more memory-efficient
- Works well for rule-based reward (exact match) on reasoning tasks

### Reward Function

```python
import re

def reward_fn(completion: str, ground_truth: str) -> float:
    """
    Returns reward given a model completion and the ground truth answer.
    """
    # Try to extract \boxed{} content
    match = re.search(r'\\boxed\{([^}]+)\}', completion)
    
    if not match:
        return -1.0  # Penalize: no boxed answer at all

    predicted = match.group(1).strip()

    # Exact string match
    if predicted == ground_truth:
        return 1.0

    # Numerical tolerance match (relative ±1e-3)
    try:
        pred_f = float(predicted)
        gt_f = float(ground_truth)
        if abs(pred_f - gt_f) / max(abs(gt_f), 1.0) < 1e-3:
            return 1.0
    except (ValueError, TypeError):
        pass

    return -0.5  # Wrong answer penalty
```

### Additional Reward Shaping (optional)
- Small `+0.1` reward if `\boxed{}` is present (format compliance bonus)
- Small `-0.1` reward if response exceeds 6000 tokens (efficiency penalty)

> [!WARNING]
> GRPO on a 30B model is slow. Start with `num_generations=4`
> (GRPO group size) and `per_device_train_batch_size=1`. Monitor GPU memory carefully.
> If OOM occurs, reduce `max_seq_length` to 2048 for RL training.

---

## Notebook / File Structure

```
nemotron_reasoning_challenge/
├── IMPLEMENTATION_PLAN.md           ← this file
├── competition_description.md
├── data/
│   ├── train.csv                    ← downloaded from Kaggle
│   ├── test.csv
│   └── cot_train.json               ← generated in phase2_cot_generation.ipynb
├── notebooks/
│   ├── phase0_eda.ipynb             ← no GPU, EDA only
│   ├── phase1_baseline_sft.ipynb    ← quick baseline, submit fast
│   ├── phase2_proxy_sft.ipynb       ← Qwen2.5-7B pipeline validation
│   ├── phase2_cot_generation.ipynb  ← Gemini API CoT generation
│   ├── phase2_sft.ipynb             ← Nemotron-30B SFT with CoT
│   └── phase3_grpo.ipynb            ← GRPO RL (bonus phase)
├── adapters/
│   ├── phase1_adapter/              ← saved LoRA adapter from Phase 1
│   └── phase2_adapter/              ← saved LoRA adapter from Phase 2
└── submissions/
    ├── submission_phase1.zip
    └── submission_phase2.zip
```

> [!IMPORTANT]
> Always save adapters to a Kaggle Dataset (not just notebook output) so they persist.
> Use `model.save_pretrained("./adapters/phaseX_adapter")` → upload as Kaggle Dataset.

---

## Verification Plan

### After Phase 0
- [ ] Know total training example count
- [ ] Have a category breakdown of puzzle types
- [ ] Manually verified 5+ examples per category

### After Phase 1
- [ ] Submission zip uploads to Kaggle without error
- [ ] `adapter_config.json` is present in the zip
- [ ] Real leaderboard score obtained (sets our floor)

### After Phase 2 (Proxy)
- [ ] Qwen2.5-7B achieves better accuracy than zero-shot baseline on held-out split
- [ ] `\boxed{}` output format is consistent

### After Phase 2 (30B SFT)
- [ ] Evaluate on local held-out 20% split of `train.csv`
- [ ] Accuracy improves vs. Phase 1 baseline
- [ ] Both Variant A (no augmentation) and Variant B (with augmentation) submitted
- [ ] Best variant identified

### After Phase 3 (GRPO)
- [ ] Reward mean is increasing over training steps (not flat or decreasing)
- [ ] No OOM errors during training
- [ ] Compare accuracy vs. Phase 2 SFT checkpoint

---

## Timeline (1–2 weeks)

| Day | Activity |
|---|---|
| Day 1 | Phase 0: EDA notebook — understand the data |
| Day 2 | Phase 1: Baseline SFT on Nemotron-30B → first submission |
| Day 3–4 | Phase 2a: Proxy model (Qwen2.5-7B) pipeline validation |
| Day 4–5 | Phase 2b: CoT generation with Gemini (runs offline, slow) |
| Day 6–7 | Phase 2c: 30B SFT with CoT → submit Variant A |
| Day 8 | Phase 2d: Augmented variant → submit Variant B |
| Day 9–12 | Phase 3: GRPO (if Phase 2 is solid and time permits) |
| Day 13–14 | Write-up, documentation (required for prize eligibility) |

---

## Key Decisions Log

| Decision | Choice | Rationale |
|---|---|---|
| CoT generator (P2a) | Nemotron itself (STaR) | Offline container, no internet; model knows puzzle domain; backsolved ~85% pass rate |
| CoT generator (P2b) | Gemini free API (local WSL) | Higher quality, more diverse reasoning; runs locally overnight, no GPU cost |
| STaR adapter reuse | Phase 1 adapter for BOTH generation AND training init | Teacher knows domain; warm-start cuts epochs 3→2; quality filter prevents circular reinforcement |
| Pipeline validation | Qwen2.5-7B proxy first | Catch format/pipeline bugs before spending 30B GPU hours |
| Augmentation | Run both with/without, benchmark | Empirical — can't know without trying |
| Phase 3 compute | Local RTX 6000 | Custom PyTorch GRPO required since TRL unavailable |
| Phase 3 priority | Aspirational/bonus | Time and memory constrained |
| Code structure | Separate notebooks per phase | Easier to iterate, cleaner Kaggle sessions |
| LoRA rank | 32 (max allowed by competition) | Maximize adapter capacity |
| Training format P1 | Manual `<\|user\|>...<\|assistant\|>\\boxed{}` | Quick baseline; **wrong for evaluation** — see Challenge 6 |
| Training format P2 | `apply_chat_template(enable_thinking=True)` + `<think>cot</think>\n\\boxed{}` | Matches competition evaluator exactly |
| Local eval params P1 | Greedy, max_new_tokens=64 | Fast iteration; **wrong** — diverges 7% from LB |
| Local eval params P2 | temp=1.0, top_p=1.0, max_new_tokens=1024 + competition `verify()` | Matches evaluator; local score now tracks LB within 1-2% |
| Metric function usage | Copy-paste into notebook | Kaggle container offline; no `utils/` imports possible |
| CoT generation batching | batch=2 (generation), left-padding | Mamba O(1) state: ~2x throughput. Left-pad required for batched causal LM inference |
| Eval split size | 5% (~475 examples) | Phase 1's 20% (1900 ex) took 200 min; 5% gives ~50 min with representative signal |
| roman_numerals CoT | Skip | Already 100% accuracy; wasted generation budget |
| Checkpoint cadence | Every 200 examples | Recoverable if 10h session times out mid-generation |

---

## Open Items

> [!IMPORTANT]
> **Action required before starting:** Get a Gemini API key at [aistudio.google.com](https://aistudio.google.com).
> Have a Groq API key as backup at [console.groq.com](https://console.groq.com) (free, no CC required).

> [!NOTE]
> **Compute Resource:** RTX 6000 (Ada Generation, ~48GB VRAM)
> - Phase 1 SFT: ~1–2 hours
> - Phase 2 proxy: ~10 minutes
> - Phase 2 30B SFT: ~2-3 hours
> - Phase 3 GRPO: ~3–4 hours (if attempted)
> Total: ~7–9 hours
