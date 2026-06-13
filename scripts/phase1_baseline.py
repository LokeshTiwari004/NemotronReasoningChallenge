# =============================================================================
# phase1_baseline.py — Raw (prompt, answer) SFT Baseline
#
# IMPORTANT: This runs on Kaggle with GPU accelerator enabled.
# The Nemotron-3-Nano-30B uses a hybrid Mamba+Attention architecture,
# so we use plain TRL + PEFT (NOT Unsloth, which is transformer-only).
#
# How to use on Kaggle:
#   1. Enable GPU accelerator (T4 x2 or P100)
#   2. Add the following Kaggle datasets:
#      - metric/nemotron-3-nano-30b-a3b-bf16  (the model)
#      - nvidia-nemotron-3-reasoning-challenge  (the data)
#   3. Add the NVIDIA utility script (for mamba_ssm):
#      - ryanholbrook/nvidia-utility-script
#   4. Paste and run cell by cell
# =============================================================================


# ── CELL 1: Install dependencies ──────────────────────────────────────────────
# Run this first. Restart kernel after if needed.
import subprocess
subprocess.run([
    "pip", "install", "-q",
    "trl>=0.12.0",
    "bitsandbytes>=0.43.0",
    "peft>=0.11.0",
    "accelerate>=0.30.0",
], check=True)


# ── CELL 2: Setup mamba_ssm (required for Nemotron) ──────────────────────────
import site
import os

cutlass_pkg_path = "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/nvidia_cutlass_dsl/python_packages/"
site.addsitedir(cutlass_pkg_path)

# Verify mamba_ssm is available
import mamba_ssm
print(f"mamba_ssm version: {mamba_ssm.__version__}")


# ── CELL 3: Load model with 4-bit quantisation ────────────────────────────────
import torch
import kagglehub
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
OUTPUT_DIR = "/kaggle/working"
LORA_RANK  = 32   # max allowed by competition

# 4-bit quantisation — essential for fitting 30B on T4/P100
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("Loading model (4-bit)... this takes ~5 minutes")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
print("Model loaded!")
print(f"Model dtype: {model.dtype}")


# ── CELL 4: Apply LoRA ────────────────────────────────────────────────────────
from peft import prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=64,
    # Mamba SSM layers (in_proj, out_proj) + MLP layers (up_proj, down_proj)
    target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ── CELL 5: Prepare dataset ───────────────────────────────────────────────────
import polars as pl
from datasets import Dataset

# System prompt that matches the evaluation setup
SYSTEM_PROMPT = (
    "You are an expert logical reasoning assistant. "
    "Think step by step to solve the given puzzle. "
    "Always place your final answer inside \\boxed{} at the end."
)

def format_example(row: dict) -> dict:
    """Format a single (prompt, answer) pair into a training string."""
    # Try to use the tokenizer's chat template if available
    try:
        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": row["prompt"]},
            {"role": "assistant", "content": f"The answer is \\boxed{{{row['answer']}}}."},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        # Fallback: manual format if no chat template
        text = (
            f"<|system|>\n{SYSTEM_PROMPT}\n"
            f"<|user|>\n{row['prompt']}\n"
            f"<|assistant|>\nThe answer is \\boxed{{{row['answer']}}}."
        )
    return {"text": text}

train_df = pl.read_csv(
    '/kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv'
).to_pandas()

print(f"Total training examples: {len(train_df)}")

# Create HuggingFace Dataset
train_data = [
    format_example({"prompt": row["prompt"], "answer": row["answer"]})
    for _, row in train_df.iterrows()
]
hf_dataset = Dataset.from_list(train_data)

# Split: 90% train, 10% eval
split = hf_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset  = split["test"]

print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")
print("\nSample formatted example:")
print(train_dataset[0]["text"][:500])


# ── CELL 6: Train ─────────────────────────────────────────────────────────────
from trl import SFTTrainer, SFTConfig

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,

    # Training duration
    num_train_epochs=3,

    # Batch size — small to fit 30B on T4 (16 GB)
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,   # effective batch = 16

    # Optimiser
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    optim="paged_adamw_8bit",         # 8-bit AdamW saves VRAM

    # Sequence
    max_seq_length=2048,

    # Logging
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,

    # Precision
    bf16=True,
    fp16=False,

    # Packing short sequences together for efficiency
    packing=True,

    # Dataset text field
    dataset_text_field="text",

    report_to="none",   # disable wandb
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

print("Starting Phase 1 training (raw SFT baseline)...")
trainer.train()
print("Training complete!")


# ── CELL 7: Quick local evaluation ───────────────────────────────────────────
import re

model.eval()
correct = 0
total = min(50, len(eval_dataset))  # sample 50 from eval set

def extract_boxed(text: str) -> str:
    """Extract the content inside the last \\boxed{} in a string."""
    matches = re.findall(r'\\boxed\{([^}]+)\}', text)
    return matches[-1].strip() if matches else ""

def is_correct(pred: str, gt: str) -> bool:
    if pred == gt:
        return True
    try:
        denom = max(abs(float(gt)), 1.0)
        if abs(float(pred) - float(gt)) / denom < 1e-3:
            return True
    except ValueError:
        pass
    return False

print(f"\nEvaluating on {total} held-out examples...")
for i in range(total):
    ex = eval_dataset[i]
    # Extract just the prompt part (before the assistant turn)
    prompt_part = ex["text"].split("<|assistant|>")[0] + "<|assistant|>"

    inputs = tokenizer(prompt_part, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=1.0,
            do_sample=False,
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pred = extract_boxed(generated)

    # Get ground truth from original data
    gt = eval_dataset[i]["text"]
    gt_boxed = extract_boxed(gt)

    if is_correct(pred, gt_boxed):
        correct += 1

print(f"Local Accuracy: {correct}/{total} = {correct/total:.2%}")


# ── CELL 8: Save adapter ──────────────────────────────────────────────────────
print(f"\nSaving LoRA adapter to {OUTPUT_DIR}...")
# Save only the adapter (not the full model)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Adapter saved!")

# List what was saved
import os
for f in os.listdir(OUTPUT_DIR):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f"  {f}  ({size/1024:.1f} KB)")


# ── CELL 9: Create submission.zip ─────────────────────────────────────────────
import subprocess, os

# Make sure adapter_config.json is present (required by evaluator)
assert os.path.exists(f"{OUTPUT_DIR}/adapter_config.json"), \
    "adapter_config.json missing! The evaluator requires this file."

result = subprocess.run(
    f"cd {OUTPUT_DIR} && zip submission.zip adapter_config.json adapter_model.safetensors",
    shell=True, capture_output=True, text=True
)
print(result.stdout)
print(result.stderr)

zip_size = os.path.getsize(f"{OUTPUT_DIR}/submission.zip") / (1024**2)
print(f"\nsubmission.zip size: {zip_size:.1f} MB")
print("Ready to submit!")
