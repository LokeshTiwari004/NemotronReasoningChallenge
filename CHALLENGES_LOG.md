# Project Challenges & Learnings Log
**Project:** NVIDIA Nemotron Reasoning Challenge
**Objective:** Fine-tune Nemotron-3-Nano-30B (SSM/Mamba hybrid) for logical reasoning.

This log documents technical hurdles, architectural adaptations, and debugging experiences encountered throughout the project. It serves as a continuous record for interview discussions and post-mortems.

---

## Phase 0: Project Strategy & Approaching Large Models

### Challenge 0: First-Time Scaling to a 30B Parameter Model
- **The Issue:** This was my first time dealing with a model of this magnitude (30 Billion parameters). The sheer scale meant I couldn't just brute-force a solution or experiment freely without risking rapid Out-of-Memory (OOM) errors, massive compute bills, or days of wasted training time on a bad hypothesis. 
- **The Solution:** I developed a deliberate, multi-phased escalation strategy to build confidence incrementally:
  1. **Phase 0 (EDA):** Understand the dataset first without touching GPUs. Mapped out the prompt structures, identified that 83% of answers were single words (binary strings, math), and classified puzzles by difficulty to prioritize CoT generation.
  2. **Phase 1 (Baseline SFT):** Train a direct `prompt -> answer` LoRA adapter *without* reasoning chains. This established a real leaderboard baseline quickly and proved the training pipeline, tokenization, and Kaggle submission formatting worked.
  3. **Phase 2 (CoT SFT):** Introduce step-by-step reasoning via Gemini backsolved CoT to get the biggest accuracy jump. Tested on a smaller proxy model (Qwen2.5-7B) first to validate the pipeline before spending heavy compute on the 30B model.
  4. **Phase 3 (RL/GRPO):** Reserve reinforcement learning for the final push, only attempting it once the SFT baseline is rock solid.
- **Learning:** When working with massive models, **fail fast and fail cheap**. Establishing a fast, lightweight iteration loop (using proxy models and quick baselines) is much more important than immediately throwing compute at the hardest problem. Structure your workflows so that at any point, you always have a working, submittable artifact.

---

## Phase 1: Infrastructure & Baseline SFT

### Challenge 1: Offline Environment Constraints & Framework Transitions
- **The Issue:** The target execution environment was shifted from a cloud-based Kaggle GPU instance (with internet) to a powerful local offline workstation (RTX PRO 6000 Blackwell, 102GB VRAM). The `trl` (Transformer Reinforcement Learning) library, used for the `SFTTrainer`, was unavailable offline and could not be installed.
- **The Solution:** Transitioned the training pipeline to the native PyTorch / Hugging Face `transformers.Trainer`. 
  - Wrote custom dataset `.map()` functions to handle tokenization, padding, and truncation manually.
  - Used `DataCollatorForLanguageModeling` to handle causal language modeling masks dynamically.
- **Learning:** High-level wrapper libraries like `trl` hide significant data preprocessing complexity. Knowing how to implement the standard `Trainer` provides resilience when working in air-gapped or restricted enterprise environments.

### Challenge 2: Mamba SSM Architecture vs. Standard Tooling
- **The Issue:** Nemotron-3-Nano-30B is not a standard Transformer; it uses a hybrid Mamba State-Space Model (SSM) with Mixture of Experts (MoE) layers. Popular optimization tools like `Unsloth` were incompatible because they expect standard Attention blocks.
- **The Solution:** Bypassed Unsloth and loaded the model natively in `bfloat16` with CPU offloading for inactive MoE experts. Designed custom LoRA configurations targeting SSM-specific projections (`in_proj`, `out_proj`, `up_proj`, `down_proj`) instead of standard attention heads (`q_proj`, `k_proj`).
- **Learning:** Deeply understanding the underlying architecture (SSM vs. Transformer) is critical before applying parameter-efficient fine-tuning (PEFT). You cannot blindly apply standard LoRA configs to novel architectures.

### Challenge 3: Triton Compiler Read-Only Permission Error
- **The Issue:** The Kaggle utility scripts required for Nvidia CUTLASS were extracted to a read-only filesystem. When PyTorch/`mamba_ssm` attempted to compile kernels via Triton, the OS threw a `[Errno 13] Permission denied` because the `ptxas-blackwell` binary lacked executable (`+x`) permissions. Standard `chmod` failed due to the read-only mount.
- **The Solution:** 
  - Wrote a Python workaround script that copied the entire `triton` module to a writable temporary directory (`/kaggle/working/`).
  - Applied recursive `chmod 0o755` permissions to the copied binaries.
  - Prepended the new writable directory to `sys.path` (at index 0) *before* importing `mamba_ssm` or `torch`, forcing Python to load our executable version of Triton.
- **Learning:** Environment variables (like `TRITON_PTXAS_PATH`) are sometimes ignored if a library uses relative `__file__` lookups. Manipulating `sys.path` dynamically is a powerful technique to monkey-patch library behavior in immutable environments.

### Challenge 4: Optimizing for 102GB VRAM (Batch Size Scaling)
- **The Issue:** Training a 30B parameter model is highly memory intensive, but the RTX 6000 had 102GB of VRAM available, leading to under-utilization with initial batch settings.
- **The Solution:** Calculated activation memory scaling. Increased `BATCH_SIZE` to safely consume ~85GB VRAM, while reducing `GRAD_ACCUM` proportionally to maintain a consistent effective batch size (16). 
- **Learning:** When scaling batch sizes, ensuring deterministic sequence truncation (`truncation=True, max_length=512`) during the `.map()` phase is critical to guarantee that VRAM usage remains flat throughout the epoch and avoids sudden Out-of-Memory (OOM) spikes.

### Challenge 5: Premature Loss Plateau & Learning Rate Instability
- **The Issue:** During the initial Phase 1 training run, the loss dropped nicely from ~8.6 to 2.7 by step 50, but then abruptly spiked to 3.5 at step 60 and began oscillating without further improvement.
- **The Solution:** Diagnosed that the learning rate (`2e-4`) was too high for a 30B parameter Mamba architecture. The loss destabilization perfectly coincided with the end of the warmup phase (`warmup_ratio=0.05` on 1425 steps = ~71 steps), which is when the LR hit its peak. Stopped the run, reduced the `LEARNING_RATE` to `5e-5`, and increased the batch size to smooth out gradient updates.
- **Learning:** Large models (especially SSMs) are highly sensitive to learning rates. If loss drops quickly but then violently spikes and plateaus precisely as warmup ends, your peak learning rate is too aggressive.

---

## Phase 2: CoT Strategy & Training Design

### Challenge 6: Training Format Mismatch — Phase 1 vs Competition Evaluator (`enable_thinking=True`)
- **The Issue:** Phase 1 used a manually constructed chat template (`<|system|>...<|user|>...<|assistant|>\boxed{answer}`). The competition evaluator calls `tokenizer.apply_chat_template(..., enable_thinking=True)`, which produces a structurally different prompt and explicitly activates Nemotron-H's built-in thinking scratchpad mode. Phase 1 training was therefore in a different distribution than what the model is evaluated in — we trained the model to skip its `<think>...</think>` block entirely.
- **Evidence:** Phase 1 local eval (greedy, max_new_tokens=64): **56.1%**. Leaderboard (temperature=1.0, max_tokens=3584, enable_thinking=True): **63.0%**. The 7% gap is the model already doing implicit reasoning when given room — proving the thinking pathway works but isn't being trained.
- **The Solution:** Phase 2 training format uses `apply_chat_template(enable_thinking=True)` for both training examples and eval inference, with response format `<think>\n{cot}\n</think>\n\boxed{answer}`. Also discovered the evaluator appends `\nPlease put your final answer inside \`\boxed{}\`.` to every prompt — we don't add this in training, the evaluator adds it at test time.
- **Learning:** Always read the official metric code before designing training format. `enable_thinking`, `max_tokens`, and sampling parameters are critical signals about what the competition expects.

---

### Challenge 7: Local Evaluation Score Diverging from Leaderboard
- **The Issue:** Phase 1 local eval used greedy decoding (`do_sample=False`), `max_new_tokens=64`, and a custom `grade_answer()` regex. The competition metric uses `temperature=1.0`, `top_p=1.0`, `max_tokens=3584`, and its own `extract_final_answer()` + `verify()` functions. These are fundamentally different evaluation conditions, making local accuracy a poor proxy for LB score.
- **Concrete impact:** Our local 56.1% vs LB 63.0% — a 7% gap not because our model is bad, but because our local eval setup suppressed the model's reasoning capability.
- **The Solution:**
  1. Copy `verify()` and `extract_final_answer()` verbatim from `utils/nvidia-nemotron-metric.py` directly into the notebook (can't import from local paths in offline Kaggle container).
  2. Use `do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=1024` for local eval inference.
  3. Reduce eval split to 5% (~475 examples, ~50 min) instead of 20% (200 min) — faster iteration with representative signal.
- **Learning:** For competition models, always replicate the exact evaluation protocol locally. A mismatch between local and official eval is a hidden source of wasted compute and wrong conclusions.

---

### Challenge 8: CoT Generation Strategy — STaR vs External API, Adapter Reuse
- **The Issue:** Phase 2 requires chain-of-thought reasoning data. Two sub-decisions with non-obvious tradeoffs:
  1. **Where does CoT come from?** Options: (a) external API (Gemini/GPT-4), (b) Nemotron itself (self-distillation / STaR), (c) smaller local model.
  2. **Should Phase 1 adapter be used for CoT generation, training init, or both?**
- **Decision 1 — Dual-track approach:**
  - **Phase 2a (immediate):** Use Phase 1 adapter (Nemotron itself) to generate CoT via backsolved prompts (STaR). Runs in offline Kaggle container. Gets a second LB benchmark while Gemini generates.
  - **Phase 2b (better quality):** Use Gemini free API on local WSL machine (has internet, no GPU cost) to generate higher-quality CoT overnight. Upload as Kaggle Dataset.
  - **Why STaR first:** No internet required, no API quota, model already knows the puzzle domain. **Why Gemini second:** External model generates more diverse reasoning not constrained by the model's current knowledge.
- **Decision 2 — Phase 1 adapter for BOTH generation AND training init (STaR standard):**
  - Phase 1 adapter as teacher generates domain-aware reasoning (knows puzzle format, answer patterns).
  - Same adapter as student warm-start: only 2 training epochs needed vs 3+ from scratch.
  - **Key safeguard:** STaR quality filter — discard any CoT where `verify(answer, extract_final_answer(generated))` fails. Prevents circular reinforcement of wrong patterns. Expected ~80-90% pass rate with backsolved prompts.
  - **Alternative (Scenario B — knowledge distillation):** Phase 1 adapter as teacher only, fresh LoRA for training. Cleaner separation but needs more training epochs to relearn format. Not chosen because warm start is more efficient here.
- **Learning:** STaR (Self-Taught Reasoner, Zelikman 2022) is standard for self-play CoT: same model generates and trains, quality filter prevents degeneration. The backsolved rationalization trick (give answer, ask for reasoning) is essential — cold-solve pass rate is ~40%, backsolved is ~85%.

---

### Challenge 9: Offline Kaggle Container — No `utils/` Imports, No Internet
- **The Issue:** All training and evaluation runs inside an offline Kaggle GPU container. Local file paths like `utils/nvidia-nemotron-metric.py` are inaccessible. External APIs (Gemini, Groq) are inaccessible. `pip install` is unavailable.
- **The Solution — two rules going forward:**
  1. **No local imports.** Any helper function (e.g., `verify()`, `extract_final_answer()`, `classify_puzzle()`) must be **copy-pasted directly into the notebook cell** with a comment indicating the source file.
  2. **CoT generation runs locally.** `phase2_cot_generation.ipynb` runs on the local WSL machine (has internet, CPU only, no GPU quota consumed). Output (`cot_train.json`) is uploaded as a Kaggle Dataset and mounted at `/kaggle/input/` for training notebooks.
- **Learning:** Design the notebook pipeline around the offline constraint from the start. Separate internet-requiring work (CoT generation, dependency downloads) from GPU-requiring work (training, inference). They run on different machines.

---

*(This log will be updated as Phase 2 and 3 progress...)*
