"""
Phase 2A — Self-Generated CoT SFT (STaR Pipeline)
==================================================
Run on Kaggle RTX 6000 (offline container).
Cut-paste each section into a separate Jupyter cell.

Pipeline:
  1. Install packages  (restart kernel after)
  2. Load Phase 1 adapter as CoT teacher
  3. Generate CoT chains for train split (skip roman_numerals)
  4. STaR filter: keep only chains that produce the correct answer
  5. Build training dataset with <think>...</think>\n\boxed{answer} format
  6. Fine-tune with trl.SFTTrainer (loss only on assistant tokens)
  7. Local evaluation (competition metric)
  8. Save adapter + create submission zip
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Install packages  (restart kernel after running this cell)
# ─────────────────────────────────────────────────────────────────────────────

import subprocess, sys

def pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# Add Kaggle working dir to path first (utility script lands packages here)
sys.path.insert(0, "/kaggle/working")

# If running on Kaggle with the NVIDIA utility script already executed,
# packages are installed. Otherwise install manually:
# pip("unsloth")
# pip("trl>=0.18.2")
# pip("peft>=0.18.0")
# pip("datasets")

print("Packages ready. RESTART KERNEL now, then run Cell 2 onwards.")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Imports + Config
# ─────────────────────────────────────────────────────────────────────────────

# unsloth MUST be first import
import unsloth  # noqa: F401

import os, re, json, math, time, random
from pathlib import Path

import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR        = Path("/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge")
PHASE1_ADAPTER  = Path("/kaggle/input/datasets/lokeshvns/nemotron-phase1-adapter")
OUTPUT_DIR      = Path("/kaggle/working/phase2a_adapter")
COT_CKPT_DIR    = Path("/kaggle/working/cot_checkpoints")  # checkpoint JSON files
MODEL_PATH      = "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1"

COT_CKPT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Training hyper-params ──────────────────────────────────────────────────
LORA_R          = 32
LORA_ALPHA      = 64
LORA_DROPOUT    = 0.05
LEARNING_RATE   = 2e-5       # lower than Phase 1 (warm start)
NUM_EPOCHS      = 2          # warm start converges faster
BATCH_SIZE      = 8          # Big GPU! Make batch size big
GRAD_ACCUM      = 2          # effective batch = 16
MAX_SEQ_LEN     = 1024       # CoT chains ~400 tokens + prompt ~100 tokens
WARMUP_RATIO    = 0.05

# ── CoT generation params ──────────────────────────────────────────────────
GEN_BATCH       = 64         # GPU has 30GB free! Crank to max!
GEN_MAX_TOKENS  = 4096       # 4096 tokens to allow for massive <think> blocks
GEN_TEMP        = 0.7
GEN_TOP_P       = 0.9
CKPT_EVERY      = 200        # save checkpoint every N examples

# ── Eval params (match competition metric exactly) ─────────────────────────
EVAL_FRAC       = 0.05       # 5% ≈ 475 examples → ~50 min eval time
EVAL_MAX_TOKENS = 1024
EVAL_TEMP       = 1.0
EVAL_TOP_P      = 1.0

print("Config loaded.")
print(f"  Model path : {MODEL_PATH}")
print(f"  Phase1 adapter: {PHASE1_ADAPTER}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Competition metric functions  (copy-pasted verbatim)
# ─────────────────────────────────────────────────────────────────────────────

def extract_final_answer(text):
    """Competition metric answer extractor. Prioritises \\boxed{}, then text patterns."""
    if text is None:
        return "NOT_FOUND"
    boxed_starts = list(re.finditer(r"\\boxed\{", text))
    matches = []
    for i, m in enumerate(boxed_starts):
        start = m.end()
        end = boxed_starts[i + 1].start() if i + 1 < len(boxed_starts) else len(text)
        segment = text[start:end]
        last_brace = segment.rfind("}")
        matches.append(segment[:last_brace] if last_brace != -1 else segment)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        return non_empty[-1] if non_empty else matches[-1].strip()
    for pattern in [
        r"The final answer is:\s*([^\n]+)",
        r"Final answer is:\s*([^\n]+)",
        r"final answer\s*[:：]\s*([^\n]+)",
    ]:
        found = re.findall(pattern, text, re.IGNORECASE)
        if found:
            return found[-1].strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if nums:
        return nums[-1]
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else "NOT_FOUND"


def verify(stored_answer, predicted):
    """Competition metric grader. Exact string match or 1% numerical tolerance."""
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()
    if re.fullmatch(r"[01]+", stored_answer):   # binary: always exact
        return predicted.lower() == stored_answer.lower()
    try:
        return math.isclose(float(stored_answer), float(predicted),
                            rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored_answer.lower()


print("Metric functions defined.")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Load data + train/eval split
# ─────────────────────────────────────────────────────────────────────────────

def categorise(prompt):
    """Heuristic category from prompt text (mirrors Phase 0 EDA logic)."""
    p = prompt.lower()
    if any(k in p for k in ["binary", "bit", "xor", "0s and 1s", "zeros and ones"]):
        return "binary"
    if any(k in p for k in ["roman", "numeral"]):
        return "roman_numerals"
    if any(k in p for k in ["decimal", "convert", "unit", "meter", "gravity", "acceleration"]):
        return "decimal_math"
    if any(k in p for k in ["integer", "operator", "symbol", "operation"]):
        return "integer_math"
    words = re.findall(r"\b[a-z]{3,}\b", p)
    if sum(1 for w in words if w.isalpha()) > 30:
        return "word_sequence"
    return "other"


df = pd.read_csv(DATA_DIR / "train.csv")
df["category"] = df["prompt"].apply(categorise)

print("Category distribution:")
print(df["category"].value_counts().to_string())

# Stratified split: 95% train / 5% eval
random.seed(42)
eval_idx = set(
    df.groupby("category", group_keys=False)
      .apply(lambda g: g.sample(frac=EVAL_FRAC, random_state=42))
      .index
)
df_train = df[~df.index.isin(eval_idx)].reset_index(drop=True)
df_eval  = df[df.index.isin(eval_idx)].reset_index(drop=True)

# Exclude roman_numerals from CoT generation (already 100% — waste of budget)
df_cot_candidates = df_train[df_train["category"] != "roman_numerals"].reset_index(drop=True)

# CoT generation priority order (hardest first → most value if we time out)
CAT_PRIORITY = ["other", "binary", "integer_math", "decimal_math", "word_sequence"]
df_cot_candidates["_pri"] = df_cot_candidates["category"].map(
    {c: i for i, c in enumerate(CAT_PRIORITY)}
).fillna(len(CAT_PRIORITY))

# Sort by priority AND prompt length to minimize padding in batched generation
df_cot_candidates["prompt_len"] = df_cot_candidates["prompt"].str.len()
df_cot_candidates = df_cot_candidates.sort_values(["_pri", "prompt_len"]).reset_index(drop=True)

print(f"\nTrain size   : {len(df_train)}")
print(f"Eval size    : {len(df_eval)}")
print(f"CoT candidates (non-roman): {len(df_cot_candidates)}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — Load tokenizer + Phase 1 adapter as CoT teacher
# ─────────────────────────────────────────────────────────────────────────────

from unsloth import FastLanguageModel

print("Loading base model and tokenizer with Unsloth (16-bit, no bitsandbytes)...")
base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_PATH,
    max_seq_length = 4096,
    dtype = torch.bfloat16,
    load_in_4bit = False,
)
tokenizer.pad_token = tokenizer.eos_token

print("Loading Phase 1 LoRA adapter ...")
teacher_model = PeftModel.from_pretrained(base_model, str(PHASE1_ADAPTER))
FastLanguageModel.for_inference(teacher_model)  # Enable Unsloth 2x faster inference
teacher_model.eval()

print("Teacher (Phase 1 adapter) ready for CoT generation.")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — CoT generation helpers
# ─────────────────────────────────────────────────────────────────────────────

BACKSOLVED_SYSTEM = (
    "You are an expert logical reasoning assistant. "
    "Think step by step. Put your final answer inside \\boxed{}."
)

def make_cot_prompt(puzzle_prompt: str, correct_answer: str) -> str:
    """Backsolved rationalisation prompt: give the model the answer, ask for reasoning."""
    return (
        f"{puzzle_prompt}\n\n"
        f"The correct answer is: {correct_answer}\n\n"
        "Explain step by step HOW to arrive at this answer. Keep your explanation CONCISE and under 500 words:\n"
        "1. Identify the hidden rule from the examples.\n"
        "2. Verify the rule holds for ALL examples shown.\n"
        "3. Apply the rule to the test input to confirm the answer.\n\n"
        f"End with: \\boxed{{{correct_answer}}}"
    )


def build_generation_input(puzzle_prompt: str, correct_answer: str) -> str:
    """Format into the chat template the teacher was trained with."""
    messages = [
        {"role": "user", "content": make_cot_prompt(puzzle_prompt, correct_answer)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def generate_cot_batch(prompts: list[str]) -> list[str]:
    """Generate CoT completions for a batch of formatted prompt strings."""
    # Left-pad for batched causal LM generation
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    ).to(teacher_model.device)

    with torch.no_grad():
        out_ids = teacher_model.generate(
            **inputs,
            max_new_tokens=GEN_MAX_TOKENS,
            do_sample=True,
            temperature=GEN_TEMP,
            top_p=GEN_TOP_P,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (strip the prompt prefix)
    generated = []
    for i, ids in enumerate(out_ids):
        new_ids = ids[inputs["input_ids"].shape[1]:]
        generated.append(tokenizer.decode(new_ids, skip_special_tokens=True))

    tokenizer.padding_side = "right"  # reset for training
    return generated


def extract_think_block(raw_completion: str) -> str | None:
    """Pull the <think>...</think> content out of the completion."""
    m = re.search(r"<think>(.*?)</think>", raw_completion, re.DOTALL)
    return m.group(1).strip() if m else None


print("CoT generation helpers defined.")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — STaR CoT generation loop  (the slow overnight step)
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    """Merge all existing checkpoint files into one id→record dict."""
    records = {}
    for f in sorted(COT_CKPT_DIR.glob("ckpt_*.json")):
        with open(f) as fp:
            for rec in json.load(fp):
                records[rec["id"]] = rec
    return records


def save_checkpoint(batch_records: list, batch_idx: int):
    path = COT_CKPT_DIR / f"ckpt_{batch_idx:06d}.json"
    with open(path, "w") as fp:
        json.dump(batch_records, fp, indent=2)


# Resume from checkpoint
existing = load_checkpoint()
done_ids  = set(existing.keys())
remaining = df_cot_candidates[~df_cot_candidates["id"].astype(str).isin(done_ids)]

print(f"Already generated : {len(done_ids)}")
print(f"Remaining         : {len(remaining)}")

kept_new, rejected_new = 0, 0
batch_records: list = []

rows = remaining.to_dict("records")
for batch_start in range(0, len(rows), GEN_BATCH):
    batch = rows[batch_start : batch_start + GEN_BATCH]

    # Build prompts
    formatted = [build_generation_input(r["prompt"], r["answer"]) for r in batch]

    try:
        completions = generate_cot_batch(formatted)
    except Exception as e:
        print(f"  [ERROR] batch {batch_start}: {e} — skipping")
        continue

    for row, completion in zip(batch, completions):
        predicted = extract_final_answer(completion)
        passed    = verify(row["answer"], predicted)

        if passed:
            think_block = extract_think_block(completion)
            # Fall back to the full completion if no <think> tags (some models skip them)
            cot_text = think_block if think_block else completion.strip()
            batch_records.append({
                "id"      : str(row["id"]),
                "prompt"  : row["prompt"],
                "answer"  : row["answer"],
                "category": row["category"],
                "cot"     : cot_text,
            })
            kept_new += 1
        else:
            if rejected_new == 0:
                print("\n--- CAVEMAN DEBUG: WHY MODEL FAIL? ---")
                print(f"PROMPT (end): {row['prompt'][-200:]}")
                print(f"CORRECT ANSWER WAS: {row['answer']}")
                print(f"MODEL SAID:\n{completion}")
                print(f"WE EXTRACTED: {predicted}")
                print("--------------------------------------\n")
            rejected_new += 1

    # Checkpoint every CKPT_EVERY examples processed
    total_processed = batch_start + len(batch)
    if total_processed % CKPT_EVERY < GEN_BATCH or batch_start + GEN_BATCH >= len(rows):
        if batch_records:
            save_checkpoint(batch_records, batch_start)
            print(
                f"  [CKPT] {total_processed}/{len(rows)} processed | "
                f"kept={kept_new} rejected={rejected_new} "
                f"keep_rate={kept_new/(kept_new+rejected_new+1e-9)*100:.1f}%"
            )
            batch_records = []

# Final merge across all checkpoints
all_cot = list(load_checkpoint().values())
all_cot.extend(existing.values())
# deduplicate
cot_by_id = {r["id"]: r for r in all_cot}
cot_records = list(cot_by_id.values())

print(f"\n── CoT Generation Summary ──")
print(f"  Total generated + kept : {len(cot_records)}")
print(f"  New kept               : {kept_new}")
print(f"  New rejected           : {rejected_new}")
if kept_new + rejected_new > 0:
    print(f"  Keep rate (this run)   : {kept_new/(kept_new+rejected_new)*100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 — Build training dataset
# ─────────────────────────────────────────────────────────────────────────────

# roman_numerals rows use the plain \boxed{answer} format (no CoT)
roman_rows = df_train[df_train["category"] == "roman_numerals"].to_dict("records")

def format_cot_example(puzzle_prompt: str, cot: str, answer: str) -> str:
    """Build a full training string using the tokenizer chat template."""
    assistant_content = f"<think>\n{cot}\n</think>\n\\boxed{{{answer}}}"
    return tokenizer.apply_chat_template(
        [
            {"role": "user",      "content": puzzle_prompt},
            {"role": "assistant", "content": assistant_content},
        ],
        tokenize=False,
        enable_thinking=True,
    )


def format_plain_example(puzzle_prompt: str, answer: str) -> str:
    """Plain \boxed{answer} format for roman_numerals (already 100% accurate)."""
    return tokenizer.apply_chat_template(
        [
            {"role": "user",      "content": puzzle_prompt},
            {"role": "assistant", "content": f"\\boxed{{{answer}}}"},
        ],
        tokenize=False,
        enable_thinking=True,
    )


train_texts = []

# CoT examples
for rec in cot_records:
    train_texts.append(format_cot_example(rec["prompt"], rec["cot"], rec["answer"]))

# Roman numeral examples (no CoT needed)
for rec in roman_rows:
    train_texts.append(format_plain_example(rec["prompt"], rec["answer"]))

random.shuffle(train_texts)

print(f"Training examples total : {len(train_texts)}")
print(f"  CoT examples          : {len(cot_records)}")
print(f"  Plain (roman)         : {len(roman_rows)}")
print("\nSample (first 200 chars):")
print(train_texts[0][:200])

hf_dataset = Dataset.from_dict({"text": train_texts})


# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 — Attach fresh LoRA adapter for Phase 2A training
# ─────────────────────────────────────────────────────────────────────────────

# Merge Phase 1 weights into the base model, then add a fresh LoRA on top.
# This gives a warm-started base that the new LoRA refines.
print("Merging Phase 1 adapter into base weights ...")
merged_model = teacher_model.merge_and_unload()

# Unsloth's optimized PEFT model setup
model = FastLanguageModel.get_peft_model(
    merged_model,
    r = LORA_R,
    target_modules = ["in_proj", "out_proj", "up_proj", "down_proj"],
    lora_alpha = LORA_ALPHA,
    lora_dropout = LORA_DROPOUT,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
)
model.print_trainable_parameters()

# Right-pad for training (standard for SFT)
tokenizer.padding_side = "right"


# ─────────────────────────────────────────────────────────────────────────────
# CELL 10 — Train with trl.SFTTrainer
# ─────────────────────────────────────────────────────────────────────────────

# No custom collator — SFTTrainer defaults to full-sequence loss.
# Prompts are ~100 tokens; CoT chains are ~400 tokens, so the prompt
# contributes only ~20% of the loss signal. Good enough for fine-tuning.

training_args = SFTConfig(
    output_dir              = str(OUTPUT_DIR),
    num_train_epochs        = NUM_EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM,
    learning_rate           = LEARNING_RATE,
    lr_scheduler_type       = "cosine",
    warmup_ratio            = WARMUP_RATIO,
    bf16                    = True,
    logging_steps           = 10,
    save_strategy           = "epoch",
    save_total_limit        = 2,
    max_seq_length          = MAX_SEQ_LEN,
    dataset_text_field      = "text",
    report_to               = "none",
    dataloader_num_workers  = 0,
)

trainer = SFTTrainer(
    model           = model,
    args            = training_args,
    train_dataset   = hf_dataset,
    tokenizer       = tokenizer,
)

print("Starting Phase 2A SFT training ...")
train_result = trainer.train()
print(f"Training complete. Loss = {train_result.training_loss:.4f}")

# Save adapter
trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
print(f"Adapter saved to {OUTPUT_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 11 — Local evaluation  (competition metric, 5% eval split)
# ─────────────────────────────────────────────────────────────────────────────

model.eval()

# Use competition eval parameters (NOT greedy like Phase 1)
EVAL_GEN_KWARGS = dict(
    max_new_tokens = EVAL_MAX_TOKENS,
    do_sample      = True,
    temperature    = EVAL_TEMP,
    top_p          = EVAL_TOP_P,
    pad_token_id   = tokenizer.eos_token_id,
)


def evaluate_example(puzzle_prompt: str, correct_answer: str) -> tuple[bool, str]:
    """Runs inference on one example, returns (correct, predicted_answer)."""
    # Match competition evaluator's prompt exactly
    eval_prompt = puzzle_prompt + "\nPlease put your final answer inside `\\boxed{}`."
    messages = [{"role": "user", "content": eval_prompt}]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, **EVAL_GEN_KWARGS)
    completion = tokenizer.decode(
        out_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    predicted = extract_final_answer(completion)
    return verify(correct_answer, predicted), predicted


# Evaluate
results_by_cat: dict[str, list[bool]] = {}
total_correct, total = 0, 0

print(f"Evaluating on {len(df_eval)} examples ...")
for i, row in df_eval.iterrows():
    correct, predicted = evaluate_example(row["prompt"], row["answer"])

    cat = row["category"]
    results_by_cat.setdefault(cat, []).append(correct)

    total_correct += int(correct)
    total += 1

    if (i + 1) % 50 == 0:
        print(f"  [{i+1}/{len(df_eval)}] running accuracy: {total_correct/total:.4f}")

# ── Results table ──────────────────────────────────────────────────────────
print("\n── Phase 2A Local Evaluation Results ──")
print(f"{'Category':<20} {'Correct':>8} {'Total':>6} {'Accuracy':>9}")
print("-" * 50)

PHASE1_SCORES = {
    "roman_numerals": 1.0000,
    "word_sequence" : 0.6466,
    "decimal_math"  : 0.5217,
    "binary"        : 0.4857,
    "integer_math"  : 0.4000,
    "other"         : 0.0298,
}

for cat in ["roman_numerals", "word_sequence", "decimal_math", "binary", "integer_math", "other"]:
    vals = results_by_cat.get(cat, [])
    if not vals:
        continue
    acc = sum(vals) / len(vals)
    delta = acc - PHASE1_SCORES.get(cat, 0)
    delta_str = f"({delta:+.4f})" if cat in PHASE1_SCORES else ""
    print(f"  {cat:<18} {sum(vals):>8} {len(vals):>6} {acc:>9.4f}  {delta_str}")

overall = total_correct / total if total else 0
print("-" * 50)
print(f"  {'OVERALL':<18} {total_correct:>8} {total:>6} {overall:>9.4f}")
print(f"\nPhase 1 baseline: 0.5611 | Phase 2A: {overall:.4f} | Δ = {overall-0.5611:+.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 12 — Save adapter + create submission zip
# ─────────────────────────────────────────────────────────────────────────────

import shutil, zipfile

SUBMISSION_DIR = Path("/kaggle/working/submission_phase2a")
SUBMISSION_DIR.mkdir(exist_ok=True)

# Copy adapter files into submission dir
shutil.copytree(str(OUTPUT_DIR), str(SUBMISSION_DIR / "adapter"), dirs_exist_ok=True)

# Zip the adapter
ZIP_PATH = Path("/kaggle/working/submission_phase2a.zip")
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in (SUBMISSION_DIR / "adapter").rglob("*"):
        if f.is_file():
            zf.write(f, f.relative_to(SUBMISSION_DIR / "adapter"))

size_mb = ZIP_PATH.stat().st_size / 1e6
print(f"Submission zip created: {ZIP_PATH}  ({size_mb:.1f} MB)")
print("Verify adapter_config.json is present:")
with zipfile.ZipFile(ZIP_PATH) as zf:
    names = zf.namelist()
    print("  " + "\n  ".join(names[:10]))
    assert "adapter_config.json" in names, "ERROR: adapter_config.json missing from zip!"
    print("  ✅ adapter_config.json present")

# ── Also save the raw CoT dataset for reuse in Phase 2B training ───────────
COT_SAVE_PATH = Path("/kaggle/working/cot_train_phase2a.json")
with open(COT_SAVE_PATH, "w") as fp:
    json.dump(cot_records, fp, indent=2)
print(f"\nCoT dataset saved: {COT_SAVE_PATH}  ({len(cot_records)} records)")
print("\nDone. Upload both files to a Kaggle Dataset for persistence.")
