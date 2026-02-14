# LLM Engineering & Optimization Lab

This repository contains R&D projects focusing on the lifecycle of Large Language Models: from architectural design to efficient fine-tuning and safety alignment.

Author: Daniil Pushkar
Role: Product Lead
Focus: Bridging the gap between deep technical research and business value in AI products.

---

## 🛠 Projects Overview

### 1. [Transformer Architecture & BPE Implementation](./01-transformer-architecture/transformer-bpe-impl.ipynb)
Domain: Architecture & NLP Core
Deep dive into the "engine" of modern NLP. Implementation of core components from scratch to understand inference costs.
 Tech: PyTorch, Custom BPE Tokenizer.
 Result: Trained a generative model from scratch, validating the impact of tokenization on context window efficiency.

### 2. [Alignment Strategies: DPO vs PPO](./02-llm-alignment-research/dpo-vs-ppo-alignment.ipynb)
Domain: AI Safety & Alignment
Research on aligning LLMs with human preferences to reduce hallucinations.
 Tech: TRL (Transformer Reinforcement Learning), PyTorch.
 Result: Compared classical RLHF (PPO) with modern DPO, demonstrating DPO's advantage in computational efficiency for business adoption.

### 3. [Efficient Fine-Tuning (PEFT/LoRA)](./03-peft-optimization/lora-dora-finetuning.ipynb)
Domain: Resource Optimization
Fine-tuning heavy models on limited hardware using Low-Rank Adapters.
 Tech: Hugging Face PEFT, BitsAndBytes (Quantization).
 Result: Successfully adapted an LLM for specific domain tasks using <10% of trainable parameters, proving the viability of on-premise deployment on cheaper hardware.

---

## 💻 Tech Stack
 Frameworks: PyTorch, Hugging Face (Transformers, PEFT, TRL).
 Optimization: BitsAndBytes (4-bit/8-bit quantization), LoRA/DoRA.
 Environment: Google Colab (T4 GPU), Linux.
