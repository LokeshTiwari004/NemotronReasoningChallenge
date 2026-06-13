# =============================================================================
# phase2_generate_cot.py — Generate Chain-of-Thought Reasoning with Gemini
#
# Run this LOCALLY (or on a Kaggle CPU notebook) BEFORE training.
# It calls the Gemini free API to generate step-by-step reasoning for
# every training example, then saves a cot_dataset.json file that
# you attach to your Kaggle training notebook as a dataset.
#
# Setup:
#   pip install google-genai
#   Set your GEMINI_API_KEY environment variable (get one free at
#   https://aistudio.google.com/app/apikey)
#
# Runtime estimate:
#   ~1000 examples × 4 sec/call ≈ ~70 minutes on free tier (15 RPM limit)
# =============================================================================

import os
import json
import time
import polars as pl
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "YOUR_KEY_HERE")
GEMINI_MODEL    = "gemini-2.0-flash"   # Free tier model (fast + smart)
OUTPUT_FILE     = "cot_dataset.json"
CHECKPOINT_FILE = "cot_checkpoint.json"   # Resume from here if interrupted
SLEEP_SECONDS   = 4.1                  # 60/15 RPM = 4 sec + buffer
MAX_RETRIES     = 3

# ── Load training data ────────────────────────────────────────────────────────
# Update this path if running locally (not on Kaggle)
DATA_PATH = "train.csv"  # or '/kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv'

train = pl.read_csv(DATA_PATH)
print(f"Loaded {len(train):,} training examples")

# ── Load checkpoint (resume support) ──────────────────────────────────────────
if Path(CHECKPOINT_FILE).exists():
    with open(CHECKPOINT_FILE) as f:
        completed = json.load(f)
    print(f"Resuming from checkpoint: {len(completed)} already done")
else:
    completed = {}

# ── Gemini client ─────────────────────────────────────────────────────────────
from google import genai
from google.genai import types

client = genai.Client(api_key=GEMINI_API_KEY)

# ── CoT generation prompt ─────────────────────────────────────────────────────
COT_SYSTEM = (
    "You are an expert at logical reasoning. "
    "When given a puzzle and its correct answer, you explain the step-by-step reasoning "
    "process that leads to that answer. Be clear, concise, and educational."
)

def build_cot_prompt(puzzle: str, answer: str) -> str:
    return f"""Here is a logical reasoning puzzle and its correct answer.

PUZZLE:
{puzzle}

CORRECT ANSWER: {answer}

Please provide a clear step-by-step explanation of how to solve this puzzle and arrive at the answer "{answer}".
Your explanation should:
1. Identify the type of transformation/rule being used
2. Apply the rule step by step
3. Confirm the final answer matches

Write the explanation in a natural, flowing style. End with: "Therefore, the answer is {answer}."
"""

# ── Main generation loop ───────────────────────────────────────────────────────
results = list(completed.values())

for i, row in enumerate(train.iter_rows(named=True)):
    puzzle_id = row["id"]

    # Skip already processed
    if puzzle_id in completed:
        continue

    prompt_text = row["prompt"]
    answer      = row["answer"]

    cot_text = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=build_cot_prompt(prompt_text, answer),
                config=types.GenerateContentConfig(
                    system_instruction=COT_SYSTEM,
                    temperature=0.3,
                    max_output_tokens=1024,
                ),
            )
            cot_text = response.text.strip()
            break
        except Exception as e:
            print(f"  [attempt {attempt+1}] Error for {puzzle_id}: {e}")
            time.sleep(10 * (attempt + 1))  # exponential backoff

    if cot_text is None:
        print(f"  FAILED after {MAX_RETRIES} attempts for {puzzle_id}, skipping")
        cot_text = f"The answer is {answer}."  # fallback: just the answer

    entry = {
        "id":     puzzle_id,
        "prompt": prompt_text,
        "cot":    cot_text,
        "answer": answer,
    }
    results.append(entry)
    completed[puzzle_id] = entry

    # Save checkpoint every 50 examples
    if (i + 1) % 50 == 0:
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(completed, f, indent=2)
        print(f"  [{i+1}/{len(train)}] Checkpoint saved. Last: {puzzle_id}")

    time.sleep(SLEEP_SECONDS)

# ── Save final dataset ────────────────────────────────────────────────────────
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone! {len(results):,} examples saved to {OUTPUT_FILE}")
print(f"File size: {Path(OUTPUT_FILE).stat().st_size / 1024:.0f} KB")

# ── Preview ───────────────────────────────────────────────────────────────────
print("\n=== Sample CoT entry ===")
sample = results[0]
print(f"ID     : {sample['id']}")
print(f"Answer : {sample['answer']}")
print(f"CoT    :\n{sample['cot'][:600]}")
