# =============================================================================
# phase2_sft_cot.py — SFT with Chain-of-Thought + Public Data Augmentation
#
# IMPORTANT: Run on Kaggle with GPU enabled (T4 x2 or P100).
#
# Prerequisites:
#   1. Run phase2_generate_cot.py locally first to produce cot_dataset.json
#   2. Upload cot_dataset.json to Kaggle as a private dataset
#      (Kaggle → Datasets → New Dataset → upload cot_dataset.json)
#   3. Attach that dataset to this notebook
#
# This script trains a much better model than Phase 1 because:
#   - The model learns step-by-step reasoning (not just final answers)
#   - Augmented with NuminaMath and OpenMathInstruct public datasets
# =============================================================================


# ── CELL 1: Install dependencies ──────────────────────────────────────────────
import subprocess
subprocess.run([
    "pip", "install", "-q",
    "trl>=0.12.0",
    "bitsandbytes>=0.43.0",
    "peft>=0.11.0",
    "accelerate>=0.30.0",
    "datasets>=2.19.0",
], check=True)


# ── CELL 2: Setup mamba_ssm ───────────────────────────────────────────────────
import site
site.addsitedir(
    "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/"
    "nvidia_cutlass_dsl/python_packages/"
)
import mamba_ssm
print(f"mamba_ssm OK: {mamba_ssm.__version__}")


# ── CELL 3: Load model + tokenizer ───────────────────────────────────────────
import torch
import kagglehub
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
OUTPUT_DIR = "/kaggle/working"
LORA_RANK  = 32

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

print("Loading model (4-bit)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=64,
    target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ── CELL 4: Load and format the CoT dataset ───────────────────────────────────
import json
from datasets import Dataset, concatenate_datasets

SYSTEM_PROMPT = (
    "You are an expert logical reasoning assistant. "
    "Think step by step to solve the given puzzle. "
    "Always place your final answer inside \\boxed{} at the end."
)

# ---- 4a: Competition CoT data ------------------------------------------------
# Update this path to where you attached your cot_dataset.json
COT_PATH = "/kaggle/input/nemotron-cot-dataset/cot_dataset.json"

with open(COT_PATH) as f:
    cot_data = json.load(f)

print(f"Loaded {len(cot_data):,} CoT examples")

def format_cot_example(entry: dict) -> dict:
    """Format a CoT entry: reasoning chain + boxed answer."""
    assistant_response = (
        f"{entry['cot']}\n\n"
        f"Therefore, the final answer is $\\boxed{{{entry['answer']}}}$."
    )
    try:
        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": entry["prompt"]},
            {"role": "assistant", "content": assistant_response},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        text = (
            f"<|system|>\n{SYSTEM_PROMPT}\n"
            f"<|user|>\n{entry['prompt']}\n"
            f"<|assistant|>\n{assistant_response}"
        )
    return {"text": text, "source": "competition"}

competition_ds = Dataset.from_list([format_cot_example(e) for e in cot_data])
print(f"Competition dataset: {len(competition_ds):,} examples")


# ---- 4b: Public augmentation — NuminaMath (NVIDIA-recommended) ---------------
try:
    from datasets import load_dataset
    numina = load_dataset("AI-MO/NuminaMath-CoT", split="train")

    def format_numina(ex):
        # NuminaMath has 'problem' and 'solution' fields
        try:
            messages = [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": ex["problem"]},
                {"role": "assistant", "content": ex["solution"]},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            text = (
                f"<|system|>\n{SYSTEM_PROMPT}\n"
                f"<|user|>\n{ex['problem']}\n"
                f"<|assistant|>\n{ex['solution']}"
            )
        return {"text": text, "source": "numina"}

    # Sample 3000 examples to avoid over-weighting public data
    numina_sample = numina.shuffle(seed=42).select(range(min(3000, len(numina))))
    numina_ds = numina_sample.map(format_numina, remove_columns=numina.column_names)
    print(f"NuminaMath augmentation: {len(numina_ds):,} examples")
except Exception as e:
    print(f"NuminaMath unavailable: {e} — skipping augmentation")
    numina_ds = None


# ---- 4c: Combine datasets with 70/30 ratio ----------------------------------
if numina_ds is not None:
    # 70% competition, 30% public
    n_public = min(len(numina_ds), int(len(competition_ds) * 0.43))
    public_subset = numina_ds.shuffle(seed=42).select(range(n_public))
    combined = concatenate_datasets([competition_ds, public_subset])
else:
    combined = competition_ds

combined = combined.shuffle(seed=42)
split    = combined.train_test_split(test_size=0.05, seed=42)
train_ds = split["train"]
eval_ds  = split["test"]

print(f"\nFinal dataset — Train: {len(train_ds):,} | Eval: {len(eval_ds):,}")
print("\nSample (first 400 chars):")
print(train_ds[0]["text"][:400])


# ── CELL 5: Train ─────────────────────────────────────────────────────────────
from trl import SFTTrainer, SFTConfig

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,

    # With CoT the sequences are longer → fewer epochs needed
    num_train_epochs=2,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,

    learning_rate=1e-4,           # slightly lower LR for CoT (longer sequences)
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    optim="paged_adamw_8bit",

    max_seq_length=4096,           # CoT responses are longer

    logging_steps=10,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    load_best_model_at_end=True,

    bf16=True,
    fp16=False,
    packing=False,                 # disable packing for variable-length CoT

    dataset_text_field="text",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
)

print("Starting Phase 2 SFT with Chain-of-Thought...")
trainer.train()
print("Training complete!")


# ── CELL 6: Local evaluation ──────────────────────────────────────────────────
import re

def extract_boxed(text: str) -> str:
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

model.eval()
correct = 0
total   = min(50, len(eval_ds))

print(f"\nEvaluating on {total} held-out examples...")
for i in range(total):
    ex_text = eval_ds[i]["text"]
    try:
        prompt_part = ex_text.split("<|assistant|>")[0] + "<|assistant|>"
    except Exception:
        continue

    inputs = tokenizer(prompt_part, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=1.0,
            do_sample=False,
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pred  = extract_boxed(generated)
    gt    = extract_boxed(ex_text)

    if is_correct(pred, gt):
        correct += 1

print(f"Phase 2 Local Accuracy: {correct}/{total} = {correct/total:.2%}")


# ── CELL 7: Save adapter and zip ──────────────────────────────────────────────
import os, subprocess

print(f"\nSaving adapter to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

assert os.path.exists(f"{OUTPUT_DIR}/adapter_config.json"), \
    "adapter_config.json missing!"

result = subprocess.run(
    f"cd {OUTPUT_DIR} && zip submission.zip adapter_config.json adapter_model.safetensors",
    shell=True, capture_output=True, text=True
)
print(result.stdout or result.stderr)

zip_size = os.path.getsize(f"{OUTPUT_DIR}/submission.zip") / (1024**2)
print(f"submission.zip: {zip_size:.1f} MB — ready to submit!")
