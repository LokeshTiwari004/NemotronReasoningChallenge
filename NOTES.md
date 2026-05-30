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
| H1 | Roman numerals will be solved well even by Phase 1 baseline (no CoT) | >90% accuracy on `roman_numerals` in Phase 1 | ❓ |
| H2 | Decimal math is learnable with simple CoT (find ratio → apply) | Significant Phase 2 improvement over Phase 1 | ❓ |
| H3 | Binary puzzles benefit most from chain-of-thought | Largest delta between Phase 1 and Phase 2 on `binary` | ❓ |
| H4 | Adding NuminaMath augmentation helps `decimal_math` / `integer_math` but hurts `binary` | Mixed effect, best to benchmark both | ❓ |
| H5 | Word sequence accuracy depends on seeing enough cipher examples in CoT | More CoT examples → better mapping reconstruction | ❓ |
| H6 | `other` category (symbol manipulation) is nearly unsolvable without fine-grained CoT | Low accuracy persists even after Phase 2 | ❓ |
| H7 | Proxy model (Qwen2.5-7B) pipeline improvement correlates with Nemotron-30B improvement | If proxy improves by X%, 30B improves similarly | ❓ |

---

## Phase 1 — Baseline SFT Results

> _Fill in after running `notebooks/phase1_baseline_sft.ipynb`_

**Date run:** ___________
**Kaggle GPU hours used:** ___________
**Training config:** rank=32, batch=2, accum=8, lr=2e-4, epochs=3

### Leaderboard Score
| Metric | Value |
|---|---|
| Overall accuracy | ___ |
| Submission ID | ___ |

### Local Eval (80/20 split)
| Category | Accuracy |
|---|---|
| `decimal_math` | ___ |
| `binary` | ___ |
| `roman_numerals` | ___ |
| `word_sequence` | ___ |
| `integer_math` | ___ |
| `other` | ___ |
| **Overall** | ___ |

### Observations
- _Write notes here after running_

### Hypotheses Updated
- _Did H1 (roman numerals easy) hold? etc._

---

## Phase 2 — CoT SFT Results

> _Fill in after running `notebooks/phase2_sft.ipynb`_

**Date run:** ___________
**CoT generation model:** ___ (Gemini / Groq / OpenAI)
**CoT examples generated:** _____ / 9500
**Kaggle GPU hours used:** ___________

### Leaderboard Score
| Variant | Score | Delta vs Phase 1 |
|---|---|---|
| Variant A (no augmentation) | ___ | ___ |
| Variant B (+ public datasets) | ___ | ___ |

### Local Eval (80/20 split)
| Category | Phase 1 | Phase 2 Var A | Phase 2 Var B |
|---|---|---|---|
| `decimal_math` | ___ | ___ | ___ |
| `binary` | ___ | ___ | ___ |
| `roman_numerals` | ___ | ___ | ___ |
| `word_sequence` | ___ | ___ | ___ |
| `integer_math` | ___ | ___ | ___ |
| `other` | ___ | ___ | ___ |
| **Overall** | ___ | ___ | ___ |

### CoT Quality Observations
- _Were Gemini-generated CoTs correct? Did they identify the right rules?_
- _Any categories where CoT was consistently wrong?_

### Hypotheses Updated
- _Update H1-H7 here_

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

### On Kaggle Free Tier Budgeting
- 30 GPU-hours per week
- Phase 1 SFT (~3–4h), Phase 2 proxy (0.5h), Phase 2 30B SFT (~4h), Phase 3 GRPO (~5h)
- **Total estimated: ~13h** — well within 30h/week limit
- Don't forget: CPU notebooks (EDA, CoT generation) are unlimited

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
- Upgraded to local **RTX 6000 (~48GB VRAM)**. This enables higher batch sizes (e.g. `BATCH_SIZE = 4`) and much faster training epochs.
- `trl` package is unavailable in the offline environment, so we are now using the standard `transformers.Trainer` with a custom dataset `.map()` tokenization function and `DataCollatorForLanguageModeling(mlm=False)`.
- Replaced Unsloth references since the model relies on the Mamba architecture which Unsloth currently does not support.
