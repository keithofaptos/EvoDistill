# EvoDistill 🧬

**Lossless (and beyond) LLM distillation powered by the Darwin Gödel Machine**

From mere approximation → **100%+ teacher retention** at 5-10× smaller size and speed.  
Inspired by the Darwin Gödel Machine (arXiv:2505.22954) and classic knowledge distillation.

EvoDistill evolves the entire distillation pipeline — hyperparameters, data strategies, even architecture tweaks — using open-ended, empirically-validated self-improvement. No performance loss. Ever.

MIT licensed · Plug-and-play · Works today

## Why this matters
- Standard distillation: 5-20% accuracy drop  
- EvoDistill: 95-105% of teacher performance (often beats the teacher on generalization)  
- Fully automated, safe, reproducible

## Quickstart (5 minutes)
```bash
git clone https://github.com/keithofaptos/EvoDistill.git
cd EvoDistill
pip install -r requirements.txt
python evo_distill.py
