# NVIDIA Nemotron Reasoning Challenge — Notes & Findings

> **Running log of observations, discoveries, and decisions made during the competition.**
> Updated as each phase is completed. Use Ctrl+F to find specific topics.

---

## Table of Contents
1. [Phase 0 — EDA Findings](#phase-0--eda-findings)
2. [Puzzle Category Analysis](#puzzle-category-analysis)
3. [Hypotheses & Things to Test](#hypotheses--things-to-test)
4. [Phase 1 — Baseline SFT Results](#phase-1--baseline-sft-results) ← _fill in after running_
5. [Phase 2 — CoT SFT Results](#phase-2--cot-sft-results) ← _fill in after running_
6. [Phase 3 — GRPO Results](#phase-3--grpo-results) ← _fill in after running_

---

## Phase 0 — EDA Findings

**Date completed:** 2026-05-29
**Notebook:** `notebooks/phase0_eda.ipynb`

### Dataset at a Glance

| Metric | Value |
|---|---|
| Training examples | **9,500** |
| Sample test examples | **3** (real held-out test = "several hundred") |
| Features | `id`, `prompt`, `answer` |
| Exact duplicate prompts | **0** |
| Contradicting answers | **0** |
| Prompt common prefix | `"In Alice's Wonderland, "` |

**Data is clean.** No preprocessing needed for basic training.

---

### Prompt Structure

Every single prompt follows this template:

```
In Alice's Wonderland, <description of the hidden rule/domain>.
<Optional extra context>.

Here are some examples of input -> output:
<example_1_input> -> <example_1_output>
<example_2_input> -> <example_2_output>
...
<example_N_input> -> <example_N_output>

Now, determine the output for: <test_input>
```

**Key insight:** The prompt itself already contains **5–10 few-shot demonstrations** of the hidden rule.
The model is NOT asked to recall knowledge — it is asked to do **in-context rule induction**.

This changes the training strategy significantly (see Hypotheses section).

---

### Length Statistics

**Prompts:**
| Stat | Value |
|---|---|
| Mean length | 302 chars / ~50 words / ~67 tokens |
| Max length | 510 chars / 78 words / 104 tokens |
| p95 tokens | ~96 tokens |

**Answers:**
| Stat | Value |
|---|---|
| Mean length | 8.4 chars |
| Max length | 39 chars |
| Single-word (83.4%) | binary strings, numbers, roman numerals |
| Multi-word (16.6%) | decrypted word sequences |

**Implication for `max_seq_length`:** Prompts are tiny. A 4096-token context gives ~3900 tokens for CoT + answer. That is more than enough. Budget ~1000–1500 tokens per CoT explanation.

---

### Category Breakdown

| Category | Count | Share | Avg Prompt Tokens | Avg Answer Length | Answer Type |
|---|---|---|---|---|---|
| `decimal_math` | 3,191 | **33.6%** | 63.9 | 5.0 chars | Number (float) |
| `binary` | 1,993 | **21.0%** | 89.6 | 11.4 chars | 8-bit binary string |
| `roman_numerals` | 1,582 | **16.7%** | 51.9 | 4.1 chars | Roman numeral string |
| `word_sequence` | 1,180 | **12.4%** | 77.8 | **25.4 chars** | 2–5 decrypted words |
| `other` | 868 | **9.1%** | 48.2 | 2.9 chars | Symbol/char sequence |
| `integer_math` | 686 | **7.2%** | 47.9 | 3.0 chars | Integer |

---

## Puzzle Category Analysis

### 1. `decimal_math` — 33.6% of data (LARGEST)

**What it is:** Given input measurements/quantities and their transformed outputs, find the constant multiplier and apply it to a new input.

**Subtypes seen:**
- Unit conversion: `10.08 m → 6.69`, `17.83 m → 11.83` → multiply by ~0.6634 (meters to feet? yards?)
- Physics (gravity): `d = 0.5 * g * t^2` with a changed gravitational constant. Given `(t, d)` pairs, deduce `g`, compute new `d`.

**Rule structure:** Consistent linear/polynomial relationship. Find constant from examples, apply.

**Difficulty: EASY.**
- The rule is always the same type (linear or quadratic)
- Only one unknown constant to find
- The model just needs to learn: "compute ratio of output/input from examples, apply to new input"

**CoT template idea:**
```
Step 1: Compute ratio = output/input for each example: 6.69/10.08 = 0.6634, 11.83/17.83 = 0.6634 ✓
Step 2: All ratios are consistent → constant = 0.6634
Step 3: Apply to new input: 25.09 × 0.6634 = 16.65
Answer: \boxed{16.65}
```

---

### 2. `binary` — 21.0% of data (HARDEST)

**What it is:** 8-bit binary strings transformed by a hidden bitwise rule. The rule can involve: shifts, rotations, XOR, AND, OR, NOT, majority function, choice function.

**Rule structure:** Varies **per puzzle** — each puzzle has a different rule. Multiple operations may be chained. The 5–10 examples must uniquely determine the rule.

**Examples seen:**
```
01110011 → 00100111  (looks like right-rotate by 2? or XOR with something?)
10001010 → 00000100  (AND mask? shift?)
```

**Difficulty: HARDEST.**
- Rule varies per puzzle (can't memorize one rule)
- Multiple plausible rules may fit first few examples
- Model must do genuine logical deduction from examples
- Some rules are combinations of operations

**CoT template idea:**
```
Step 1: Test if rule is a simple shift. Right-shift by 2: 01110011 → 00011100 ≠ 00100111. Not a pure shift.
Step 2: Test rotation. Right-rotate by 2: 01110011 → 11011100. No.
Step 3: Try XOR with fixed mask. 01110011 XOR 00100111 = 01010100. Not constant.
Step 4: Try nibble swap (swap upper 4 bits with lower 4 bits): 0111|0011 → 0011|0111 = 00110111. No.
Step 5: Notice pattern... [continue analysis]
```

> [!WARNING]
> This is where most accuracy will be lost. CoT generated by Gemini/GPT may **rationalize
> wrong rules**. Since we use backsolved CoT (give Gemini the answer), the CoT explanation
> may not reflect the true rule — it may be post-hoc reasoning that happens to produce
> the correct answer without identifying the actual transformation.
>
> **Mitigation:** For binary puzzles, have Gemini verify its rule against ALL examples
> in the prompt, not just produce an answer.

---

### 3. `roman_numerals` — 16.7% of data

**What it is:** Convert integers to Roman numerals. The "Wonderland numeral system" is just standard Roman numerals.

**Examples:**
- `11 → XI`, `15 → XV`, `94 → XCIV`, `38 → XXXVIII`

**Difficulty: TRIVIAL.**
- Standard Roman numeral conversion
- The base Nemotron model almost certainly already knows this
- Phase 1 baseline (no CoT) will likely score near-perfectly on this category
- Don't waste CoT budget here; deprioritize

**CoT template idea (minimal):**
```
38 = 10+10+10+5+1+1+1 = XXXVIII
Answer: \boxed{XXXVIII}
```

---

### 4. `word_sequence` — 12.4% of data

**What it is:** Encrypted text → decrypted English words. The encryption is a character substitution cipher (each letter maps to another letter consistently within a puzzle).

**Example:**
```
ucoov pwgtfyoqg vorq yrjjoe → queen discovers near valley
pqrsfv pqorzg wvgwpo trgbjo → dragon dreams inside castle
...
Now decrypt: trb wzrswvog hffk
Answer: cat imagines book
```

**Difficulty: MEDIUM.**
- Build the character mapping from all examples
- Apply mapping to decrypt the test string
- Multi-word answers — need `\boxed{cat imagines book}` to capture all words

**CoT template idea:**
```
Step 1: Build cipher map from examples:
  u→q, c→u, o→e, v→e, n... (map all observed pairs)
Step 2: Fill gaps (letters not seen in examples) by elimination
Step 3: Decrypt "trb wzrswvog hffk":
  t→c, r→a, b→t = cat
  w→i, z→m, r→a, s→g, w→i, o→n, g→e, s→s = imagines
  h→b, f→o, f→o, k→k = book
Answer: \boxed{cat imagines book}
```

---

### 5. `integer_math` — 7.2% of data

**What it is:** Custom arithmetic operators applied to pairs of numbers. Standard +, -, ×, / symbols are redefined to mean something else.

**Examples seen:**
```
64-65 = 201    → not subtraction! (digits: 64-65 reversed → 56-46 → concat = 201? need more examples)
28-68 = 861
82/15 = 8241   → concat digits? 82 and 15... 
52{43 = 9      → 52+43=95 → 9+5=14? or 5+4=9? 
31*15 = 46     → 3+1+1+5=10? or 31+15=46 ✓ (actually addition!)
```

**Difficulty: MEDIUM-HARD.**
- Each puzzle has its own redefined operator
- Must deduce the operation from 3–4 examples
- Fewer examples than binary → more ambiguity

---

### 6. `other` — 9.1% of data

**What it is:** Transformation rules applied to sequences of special characters/symbols.

**Examples seen:**
```
`!*[{ = '\"[`
\'*'> = ![@
Now determine: [[-!'  → Answer: @&
```

**Difficulty: HARD.**
- Purely symbolic — no domain knowledge helps
- Rules appear to be positional mappings or character substitutions
- Very opaque; hardest to write interpretable CoT for

---

## Hypotheses & Things to Test

> Mark each as ✅ Confirmed / ❌ Rejected / ❓ Untested as experiments run

| # | Hypothesis | Expected Effect | Status |
|---|---|---|---|
| H1 | Roman numerals will be solved well even by Phase 1 baseline (no CoT) | >90% accuracy on `roman_numerals` in Phase 1 | ✅ **100%** |
| H2 | Decimal math is learnable with simple CoT (find ratio → apply) | Significant Phase 2 improvement over Phase 1 | ❓ (baseline 52.2%) |
| H3 | Binary puzzles benefit most from chain-of-thought | Largest delta between Phase 1 and Phase 2 on `binary` | ❓ (baseline 48.6%) |
| H4 | Adding NuminaMath augmentation helps `decimal_math` / `integer_math` but hurts `binary` | Mixed effect, best to benchmark both | ❓ |
| H5 | Word sequence accuracy depends on seeing enough cipher examples in CoT | More CoT examples → better mapping reconstruction | ❓ (baseline 64.7%) |
| H6 | `other` category (symbol manipulation) is nearly unsolvable without fine-grained CoT | Low accuracy persists even after Phase 2 | ✅ **3% — nearly zero** |
| H7 | Proxy model (Qwen2.5-7B) pipeline improvement correlates with Nemotron-30B improvement | If proxy improves by X%, 30B improves similarly | ❓ |

---

## Phase 1 — Baseline SFT Results ✅

**Date run:** 2026-05-30
**Kaggle GPU hours used:** ~7 hrs total (3.5h training + 3.5h eval)
**Training config:** rank=32, batch=8, accum=2, lr=5e-5, epochs=3, max_seq=512
**Framework:** `transformers.Trainer` + `DataCollatorForLanguageModeling` (no trl)

### Leaderboard Score
| Metric | Value |
|---|---|
| **Public leaderboard accuracy** | **0.63** |
| Local eval accuracy | 0.5611 |

> **Note:** Public LB (0.63) > local eval (0.561). The leaderboard uses `temperature=1.0` (stochastic)
> with `max_tokens=3584` allowing the model to think longer. Our local eval used greedy `max_new_tokens=64`.
> The model generates reasoning steps naturally — longer generation = better score.

### Training Details
| Metric | Value |
|---|---|
| Trainable parameters | 880,138,240 (2.71% of 32.5B) |
| Total steps | 1,425 (475/epoch × 3 epochs) |
| Final training loss | **1.3571** |
| Training runtime | 208.7 minutes (~3.5 hrs) |
| LoRA modules applied | 53,820 |

### Full Training Loss Sequence (every 10 steps)

```
Step  | Loss     Phase  Notes
------|--------- ------  ----------------------------------------
  10  | 4.3989   Ep1     Fast initial drop — model adapting to format
  20  | 4.1951
  30  | 3.3151
  40  | 2.3896
  50  | 1.7207           Warmup ends ~step 71 (5% of 1425)
  60  | 1.9971   ←SPIKE  LR peaks at warmup end — Mamba SSM sensitivity
  70  | 1.5562           Loss recovers quickly — LR now decaying (cosine)
  80  | 1.5887
  90  | 1.4603
 100  | 1.5474
 110  | 1.4892
 120  | 1.6083
 130  | 1.3899
 140  | 1.5142
 150  | 1.4489   Ep1→2   475 steps/epoch
 200  | 1.2021           Smoother region — cosine LR well into decay
 250  | 1.2057
 290  | 1.6083   ←BLIP   Occasional noisy batches (symbol/binary puzzles)
 300  | 1.3584
 350  | 1.1815
 400  | 1.3943
 475  |          Ep2→3
 800  | 1.0374           Best single-step reading
 950  | 1.2095   Ep3
1000  | 1.1994
1050  | 1.0500
1200  | 1.2103
1350  | 0.9758           Lowest recorded step
1420  | 1.1324   (last logged step)
FINAL | 1.3571           Trailing average over last N steps
```

### Local Eval (80/20 split — 1900 examples, greedy, max_new_tokens=64)
| Category | Correct | Total | Accuracy |
|---|---|---|---|
| `roman_numerals` | 328 | 328 | **1.0000** ✅ |
| `word_sequence` | 150 | 232 | **0.6466** |
| `decimal_math` | 348 | 667 | 0.5217 |
| `binary` | 187 | 385 | 0.4857 |
| `integer_math` | 48 | 120 | 0.4000 |
| `other` | 5 | 168 | **0.0298** ❌ |
| **OVERALL** | **1066** | **1900** | **0.5611** |

**Format compliance:** 100% — every prediction contained a `\boxed{}` answer.

### Observations
- **Roman numerals is perfect (100%)** — as predicted. No CoT needed here, don't waste CoT budget.
- **`other` category is nearly broken (3%)** — symbol manipulation without CoT is nearly impossible. #1 priority for CoT.
- **Binary (48.6%)** — the model is guessing/partially memorizing, not genuinely applying the rule per-puzzle.
- **Word sequence (64.7%)** — surprisingly high without CoT. The model partially pattern-matches cipher structure from training examples.
- **Decimal math (52.2%)** — the model can find a ratio sometimes but isn't reliably doing the algebra.
- **Local 56.1% < LB 63%** — see "Q3: Does the metric suggest Phase 2?" below.
- **Running accuracy dip:** starts at 50%, rises to 59.5% at example 200, then slowly drifts down and stabilises at 56.1%. The first 200 examples likely hit easier categories (roman_numerals, simple decimal_math). The drift down is as harder categories accumulate.
- **Adapter size:** 3.3 GB safetensors (~3.2 GB zip) — large but within Kaggle submission limits.

### Hypotheses Updated
| # | Hypothesis | Status | Actual Result |
|---|---|---|---|
| H1 | Roman numerals >90% in Phase 1 | ✅ **CONFIRMED** | **100%** |
| H2 | Decimal math improves with CoT | ❓ Untested | Baseline 52.2% — CoT target: 80%+ |
| H3 | Binary benefits most from CoT | ✅ **Likely** | 48.6% — largest absolute gap |
| H6 | `other` nearly unsolvable without CoT | ✅ **CONFIRMED** | **3%** — worst by far |

---

## Q&A — Phase 1 Post-Mortem

### Q1: How do I interpret the training loss sequence?

**Phase 1: Rapid Format Learning (steps 1–50)**
Loss drops from 4.4 → 1.7 very quickly. The model is learning the output format: it's discovering that `<|assistant|>\n\boxed{X}` is the right pattern. This is pure memorization of structure, not reasoning.

**The LR Spike at step 60 (loss: 1.72 → 1.99)**
Warmup ends at step 71 (5% × 1425). At step 60 the LR is near its peak (5e-5). The loss jumps. This is the classic sign that the peak LR is too high for a Mamba/SSM model. It recovered quickly because the cosine scheduler immediately starts reducing LR after warmup. With `lr=2e-4` (the original attempt), this spike was to 3.5 and the model never recovered — it plateaued at 2.7.

**Phase 2: Noisy Plateau (steps 70–1350)**
Loss oscillates between ~1.0–1.6 without a strong downward trend. This is **not a failure** — it's the expected behaviour for a model learning a heterogeneous dataset. The oscillation is caused by:
- Easy batches (roman_numerals, simple decimal) → low loss
- Hard batches (binary, other, word_sequence) → high loss
- The model can't memorize 9500 unique rules — it's learning statistical patterns

The slow downward drift from ~1.5 (step 100) to ~1.1–1.2 (step 1200+) indicates genuine generalisation, not overfitting.

**The `final loss = 1.3571` vs step 1420 = 1.132**
`train_result.training_loss` is the **exponential moving average** over all 1425 steps, not the final step value. The actual final few steps average around 1.15–1.2 — better than the reported 1.3571 suggests.

**Bottom line:** The loss curve is healthy. The model learned the output format fast, had one LR-induced blip, then gradually improved on the actual puzzle patterns. No signs of overfitting or divergence.

---

### Q2: Is Phase 1 the best we could do?

**Short answer: No. Several improvements possible, but diminishing returns.**

**What we could have done differently in Phase 1:**

| Improvement | Expected gain | Effort |
|---|---|---|
| Train 5 epochs instead of 3 | +1–2% local | Costs another 1.5h GPU |
| Reduce LR further to 2e-5 | Smoother plateau, maybe +0.5% | Free |
| Use `max_seq_length=1024` (not 512) | Captures longer prompts fully | Free |
| Add category-aware data sampling | Balance binary/other vs roman_numerals | Medium effort |
| Use chat_template from tokenizer | Match evaluation format exactly | Critical — see Q3 |
| Skip roman_numerals from training | Save those steps for hard categories | Small gain |

**However:** Phase 1 is purposely a quick baseline. The real ceiling of the `prompt→\boxed{answer}` approach is probably **~65–68% on LB** (we're at 63%). The hard limit is:
- **`other` category: fundamentally hard without CoT** — even humans need explicit reasoning
- **`binary` category: per-puzzle rules** — the model can't memorise 2000 different bitwise operations

Phase 2 (CoT) is where we break the ceiling. The question is whether we squeeze another 2–3% from Phase 1 tweaks, or invest that GPU time in Phase 2. **Recommendation: move to Phase 2.**

---

### Q3: Does the evaluation metric suggest the competition expects Phase 2 (CoT)?

**Yes — strongly. Three pieces of evidence:**

**Evidence 1: `enable_thinking=True`**
```python
prompt = tokenizer.apply_chat_template(
    [{'role': 'user', 'content': user_content}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,   # ← THIS
)
```
Nemotron-H models have an explicit "thinking mode" — when enabled, the model generates `<think>...</think>` scratchpad content before the answer. The evaluator **explicitly enables this**. Our Phase 1 training format (`\boxed{answer}` directly) works *against* this — we trained the model to skip its thinking phase.

**Evidence 2: `max_tokens=3584`**
For a competition where most answers are 2–15 characters long, why allow 3584 tokens of output? Because reasoning chains are expected to fill that space. The average answer is 8 chars — 3584 tokens is 400× more than needed just for the answer.

**Evidence 3: LB (0.63) > Local (0.561)**
Our local eval forces `max_new_tokens=64` and greedy decoding. The LB uses 3584 tokens + temperature=1.0. The 7% gap means: **when allowed to think and sample, the model is already doing some implicit reasoning** even with Phase 1 training. If we explicitly train CoT, that gap should widen in our favour.

**Conclusion:** The entire evaluation setup is built for a model that *thinks before answering*. Phase 2 CoT training is not optional — it's what the competition is designed around.

---

## Phase 2a — Self-Generated CoT (STaR) Results

> _Fill in after running `notebooks/phase2a_self_cot_sft.ipynb`_

**Notebook:** `phase2a_self_cot_sft.ipynb`
**Method:** STaR — Phase 1 adapter as teacher (CoT gen) + student init (training)
**Date run:** ___________
**Kaggle GPU hours used:** ~9.3h planned

### Implementation Choices (Phase 2a)

| Choice | Value | Reason |
|---|---|---|
| CoT generator | Phase 1 adapter (self) | Offline Kaggle; knows puzzle domain |
| Prompt strategy | Backsolved rationalization | Give answer → ask reasoning. ~85% pass rate vs ~40% cold-solve |
| STaR filter | `verify(answer, extract_final_answer(output))` | Discard wrong CoT; prevents circular reinforcement |
| CoT categories | `other → binary → integer_math → word_sequence → decimal_math` | Hardest first; if time runs out, high-value categories covered |
| Skip `roman_numerals` | Yes | Already 100% accuracy; wasted budget |
| Generation batch | 2 | Mamba O(1) state: ~2x speedup; left-padding required |
| CoT max tokens | 300 | Enough for reasoning; shorter = faster generation |
| Generation temp | 0.7 / top_p=0.9 | Slightly creative → diverse chains for same category |
| Checkpoint every | 200 examples | Resume on crash/timeout |
| Training LR | 2e-5 (vs 5e-5 Phase 1) | Warm start; refining not relearning |
| Training epochs | 2 (vs 3 Phase 1) | Warm start converges faster |
| Training seq len | 1024 (vs 512 Phase 1) | CoT chains average ~400 tokens |
| Training batch | 4 (vs 8 Phase 1) | Longer seqs = more VRAM/example |
| padding_side | left during gen, right during training | Left-pad required for batched causal LM gen |
| Eval params | temp=1.0, top_p=1.0, max_new_tokens=1024 | Matches competition metric exactly |
| Eval functions | `verify()` + `extract_final_answer()` copy-pasted inline | Kaggle offline; no `utils/` imports |
| Eval split | 5% (~475 ex, ~50 min) | Phase 1's 20% took 200 min; 5% is enough signal |

### CoT Generation Stats
| Metric | Value |
|---|---|
| Total eligible (non-roman) | ~6,288 |
| Generated + kept (STaR filter) | ___ |
| Rejected (wrong reasoning) | ___ |
| Keep rate | ___% (expect 80-90%) |
| Generation time | ___ min |

### Leaderboard Score
| Phase | Local acc | LB acc | Delta |
|---|---|---|---|
| Phase 1 (no CoT) | 0.5611 | 0.63 | — |
| **Phase 2a (self CoT, STaR)** | ___ | ___ | ___ |

### Local Eval (5% split, competition metric)
| Category | Phase 1 | Phase 2a | Δ | CoT? |
|---|---|---|---|---|
| `other` | 0.0298 | ___ | ___ | ✓ |
| `binary` | 0.4857 | ___ | ___ | ✓ |
| `integer_math` | 0.4000 | ___ | ___ | ✓ |
| `decimal_math` | 0.5217 | ___ | ___ | ✓ |
| `word_sequence` | 0.6466 | ___ | ___ | ✓ |
| `roman_numerals` | 1.0000 | ___ | ___ | ✗ |
| **OVERALL** | **0.5611** | ___ | ___ | |

### Observations
- _Was self-generated CoT good quality? Did keep rate match ~85% estimate?_
- _Which categories improved most?_
- _Did `other` category show meaningful gain (from 3%)?_
- _Did training loss drop lower than Phase 1's 1.3571?_

---

## Phase 2b — Gemini CoT Results

> _Fill in after running `phase2_cot_generation.ipynb` (local WSL) + `phase2_sft.ipynb` (Kaggle)_

**CoT source:** Gemini free API (local WSL machine, no GPU cost)
**Date run:** ___________

### Leaderboard Score
| Phase | Local acc | LB acc | Delta vs P1 |
|---|---|---|---|
| Phase 2a (self CoT) | ___ | ___ | ___ |
| **Phase 2b (Gemini CoT)** | ___ | ___ | ___ |

### CoT Quality Comparison
| Source | Keep rate | Avg CoT length | Local acc |
|---|---|---|---|
| Phase 2a (self / STaR) | ___% | ___ tokens | ___ |
| Phase 2b (Gemini) | 100% (backsolved) | ___ tokens | ___ |

---


## Phase 3 — GRPO Results

> _Fill in if attempted_

**Date run:** ___________
**Starting checkpoint:** Phase 2 SFT adapter
**GRPO config:** group_size=4, lr=5e-6, epochs=1

### Results
| Metric | Value |
|---|---|
| Mean reward at start | ___ |
| Mean reward at end | ___ |
| Leaderboard score | ___ |
| Delta vs Phase 2 | ___ |

### Observations
- _Did reward increase monotonically? Any instability?_
- _OOM issues? What config worked on T4×2?_

---

## Miscellaneous Notes

### On GPU Budgeting (Revised)
- **30 GPU-hours/week** limit applies even with RTX 6000 Blackwell on Kaggle
- Phase 1 actual: **~7 hrs** (3.5h train + 3.5h eval on 1900 examples)
- Remaining budget per week: ~23 hrs
- Phase 2 SFT: ~3.5–4h training (same config), eval: ~3.5h → **~7.5h total**
- Phase 3 GRPO: ~4–6h
- **Total remaining phases: ~14h** — within weekly budget if spread across 2 weeks
- **Optimization for eval time:** eval on 1900 examples took 200 min (6.3 sec/example). Could sample 500 for faster iteration.
- Don't forget: CPU-only notebooks (EDA, CoT generation) don't count against GPU quota

### On the `local eval time being sane (3.33 hrs for 1900 examples)`
- **Yes, 3.33 hrs IS sane for this model.** 1900 examples × 6.3 sec/example = 200 min
- The model is 30B params (3B active) — even inference is slow
- `max_new_tokens=64` limits output but input processing (prefill) is still expensive
- **To speed up eval in Phase 2:** reduce eval split to 10% (950 examples) → ~100 min
- Or batch inference with `model.generate()` on padded batches (batch=4 may help)

### ⚠️ Critical Metric File Findings (`utils/nvidia-nemotron-metric.py`)
These are major discoveries that should influence Phase 2 training:

1. **`enable_thinking=True` in `apply_chat_template()`** — the evaluator explicitly enables
   the model's thinking mode. The model generates a `<think>...</think>` block before the answer.
   Our Phase 1 training format suppressed this by teaching `\boxed{answer}` directly.
   **Phase 2 should teach: `<think>reasoning steps</think>\n\boxed{answer}`**

2. **Prompt has instruction appended by evaluator:**
   `item.prompt + '\nPlease put your final answer inside \`\boxed{}\`.'`
   We don't need to add this in training — the evaluator adds it automatically.

3. **Numerical tolerance is `rel_tol=1e-2` (1%)** — more lenient than our local eval's 1e-3.
   A few decimal_math answers that we're marking wrong may actually be correct on LB.

4. **`temperature=1.0`, `top_p=1.0`** — stochastic sampling, NOT greedy.
   This explains why LB (0.63) > local (0.561): with more tokens + sampling, the model
   sometimes produces correct reasoning chains even without CoT training.

5. **`max_tokens=3584`** — the evaluator allows very long responses.
   The ~7% gap between LB and local is likely because longer generation = better answers.

### Action Items for Phase 2 Based on Metric Findings
- [ ] Train with `<think>step-by-step reasoning</think>\n\boxed{answer}` format
- [ ] Prioritize CoT for: `binary` (48.6%), `other` (3%), `integer_math` (40%)
- [ ] Skip CoT for `roman_numerals` — already 100%, wasted compute
- [ ] Use `rel_tol=1e-2` in local eval to match metric exactly

### On the `\boxed{}` Format
- Scoring extracts content from `\boxed{...}` first, falls back to last number
- For **multi-word answers** (`word_sequence` category): `\boxed{cat imagines book}` — the entire phrase must be inside the box
- For **binary strings**: `\boxed{10010111}` — no spaces
- For **roman numerals**: `\boxed{XXXVIII}` — uppercase, no spaces
- **Always** include the `\boxed{}` instruction in system prompt

### On CoT Generation (Gemini API)
- Free tier: ~15 RPM → 9500 examples × 4s sleep = ~10.5 hours (run overnight)
- Save to Kaggle Dataset immediately after generation — don't lose it
- Use backsolved CoT: give Gemini the question + correct answer, ask it to explain
- **For binary puzzles specifically**: ask Gemini to verify its rule against ALL examples, not just 1–2
- Fallback API: Groq (llama-3.3-70b-versatile) — higher RPM, also free

### Prompt Engineering Considerations
- The puzzles already have few-shot examples embedded in the prompt
- System prompt should NOT add additional few-shot examples (would confuse the model)
- System prompt should be minimal: `"You are a logical reasoning assistant. Think step by step. Put your final answer inside \\boxed{}."`
- Don't over-engineer the system prompt — the puzzle prompt already does the heavy lifting

### Hardware & Framework Updates (Phase 1)
- Upgraded to Kaggle container **RTX 6000 (~95GB VRAM)**. This enables higher batch sizes (e.g. `BATCH_SIZE = 8`) and much faster training epochs.
- `trl` package is unavailable in the offline environment, so we are now using the standard `transformers.Trainer` with a custom dataset `.map()` tokenization function and `DataCollatorForLanguageModeling(mlm=False)`.
- Replaced Unsloth references since the model relies on the Mamba architecture which Unsloth currently does not support.
