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

---
*(This log will be updated as Phase 2 and 3 progress...)*
