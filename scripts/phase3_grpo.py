# =============================================================================
# phase3_grpo.py — GRPO (Reinforcement Learning) Fine-tuning
#
# IMPORTANT: Run on Kaggle with GPU enabled. This builds on Phase 2's adapter.
# GRPO (Group Relative Policy Optimization) is the same technique used in
# DeepSeek-R1 — it uses rule-based rewards to teach the model to reason better.
#
# Prerequisites:
#   - Phase 2 SFT adapter (attached as a Kaggle dataset or from /kaggle/working)
#   - trl >= 0.12.0 (has GRPOTrainer)
#
# How GRPO works:
#   1. For each puzzle, generate 4–8 candidate answers (different samples)
#   2. Score each with a reward function (correct = +1, wrong = -0.5, no \\boxed = -1)
#   3. Update the model to increase probability of high-reward answers
#   4. No separate critic model needed — memory efficient!
# =============================================================================


# ── CELL 1: Install dependencies ──────────────────────────────────────────────
import subprocess
subprocess.run([
    "pip", "install", "-q",
    "trl>=0.12.0",
    "bitsandbytes>=0.43.0",
    "peft>=0.11.0",
    "accelerate>=0.30.0",
    "vllm>=0.5.0",          # optional but speeds up generation significantly
], check=True)


# ── CELL 2: Setup mamba_ssm ───────────────────────────────────────────────────
import site
site.addsitedir(
    "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/"
    "nvidia_cutlass_dsl/python_packages/"
)
import mamba_ssm
print(f"mamba_ssm OK: {mamba_ssm.__version__}")


# ── CELL 3: Load base model + Phase 2 adapter ─────────────────────────────────
import torch
import kagglehub
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

MODEL_PATH   = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
# Update this path to your Phase 2 adapter (attached Kaggle dataset)
ADAPTER_PATH = "/kaggle/input/nemotron-phase2-adapter"
OUTPUT_DIR   = "/kaggle/working"
LORA_RANK    = 32

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

print("Loading base model (4-bit)...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

# Load Phase 2 adapter on top of the base model
print("Loading Phase 2 adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, is_trainable=True)
print("Model ready for GRPO!")
model.print_trainable_parameters()


# ── CELL 4: Reward functions (mirrors the competition metric exactly) ──────────
import re

def extract_boxed(text: str) -> str:
    """Extract content from the last \\boxed{} in text."""
    matches = re.findall(r'\\boxed\{([^}]+)\}', text)
    if matches:
        return matches[-1].strip()
    # Fallback: last number-like string
    nums = re.findall(r'\b[\d\.\-]+\b', text)
    return nums[-1] if nums else ""

def is_numerically_close(pred: str, gt: str, tol: float = 1e-3) -> bool:
    try:
        denom = max(abs(float(gt)), 1.0)
        return abs(float(pred) - float(gt)) / denom < tol
    except ValueError:
        return False

# Reward function 1: Correctness (main reward)
def reward_correctness(completions: list[str], ground_truths: list[str], **kwargs) -> list[float]:
    """
    +1.0  → correct answer (exact match or within numerical tolerance)
    -0.5  → wrong answer
    -1.0  → no \\boxed{} found
    """
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        pred = extract_boxed(completion)
        if pred == "":
            rewards.append(-1.0)
        elif pred == gt or is_numerically_close(pred, gt):
            rewards.append(1.0)
        else:
            rewards.append(-0.5)
    return rewards

# Reward function 2: Format compliance (small bonus for having \\boxed{})
def reward_format(completions: list[str], **kwargs) -> list[float]:
    """
    +0.2  → has \\boxed{} in completion
     0.0  → no \\boxed{}
    """
    return [
        0.2 if re.search(r'\\boxed\{[^}]+\}', c) else 0.0
        for c in completions
    ]

# Reward function 3: Length penalty (discourage extremely long answers)
def reward_length(completions: list[str], **kwargs) -> list[float]:
    """Small penalty for responses > 1500 tokens."""
    rewards = []
    for c in completions:
        n_tokens = len(tokenizer.encode(c))
        if n_tokens > 1500:
            rewards.append(-0.1 * min((n_tokens - 1500) / 500, 1.0))
        else:
            rewards.append(0.0)
    return rewards


# ── CELL 5: Prepare dataset for GRPO ─────────────────────────────────────────
import polars as pl
from datasets import Dataset

SYSTEM_PROMPT = (
    "You are an expert logical reasoning assistant. "
    "Think step by step to solve the given puzzle. "
    "Always place your final answer inside \\boxed{} at the end."
)

train_df = pl.read_csv(
    '/kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv'
).to_pandas()

def make_grpo_prompt(row):
    """GRPO needs the prompt formatted for generation (no answer in input)."""
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": row["prompt"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        text = (
            f"<|system|>\n{SYSTEM_PROMPT}\n"
            f"<|user|>\n{row['prompt']}\n"
            f"<|assistant|>\n"
        )
    return {"prompt": text, "ground_truth": row["answer"]}

grpo_data = [make_grpo_prompt(row) for _, row in train_df.iterrows()]
grpo_dataset = Dataset.from_list(grpo_data)

# Use full training set for GRPO
print(f"GRPO dataset: {len(grpo_dataset):,} examples")


# ── CELL 6: GRPO Training ─────────────────────────────────────────────────────
from trl import GRPOTrainer, GRPOConfig

grpo_config = GRPOConfig(
    output_dir=OUTPUT_DIR,

    # GRPO hyperparameters
    num_train_epochs=1,            # GRPO converges faster than SFT
    num_generations=4,             # generate 4 candidates per prompt
    max_new_tokens=512,            # max length per generation
    temperature=0.8,               # sampling temperature for generation

    # Training
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,            # much lower LR for RL (avoid catastrophic forgetting)
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    optim="paged_adamw_8bit",

    # KL penalty — keeps the model close to the SFT policy (stability)
    kl_coeff=0.04,

    # Logging
    logging_steps=5,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,

    bf16=True,
    fp16=False,
    report_to="none",
)

trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=grpo_dataset,
    tokenizer=tokenizer,
    reward_funcs=[
        reward_correctness,   # main reward (weighted highest automatically)
        reward_format,        # small format bonus
        reward_length,        # length penalty
    ],
)

print("Starting Phase 3 GRPO training...")
print("Watch the 'reward/mean' metric — it should increase over time.")
trainer.train()
print("GRPO training complete!")


# ── CELL 7: Save adapter ──────────────────────────────────────────────────────
import os, subprocess

print(f"\nSaving GRPO-trained adapter to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

assert os.path.exists(f"{OUTPUT_DIR}/adapter_config.json"), "adapter_config.json missing!"

result = subprocess.run(
    f"cd {OUTPUT_DIR} && zip submission.zip adapter_config.json adapter_model.safetensors",
    shell=True, capture_output=True, text=True
)
print(result.stdout or result.stderr)

zip_size = os.path.getsize(f"{OUTPUT_DIR}/submission.zip") / (1024**2)
print(f"submission.zip: {zip_size:.1f} MB — ready to submit!")
