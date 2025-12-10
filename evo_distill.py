# EvoDistill — Evolutionary Knowledge Distillation powered by Darwin Gödel Machine
# MIT License © 2025 Keith of Aptos + Grok
# github.com/keithofaptos/EvoDistill

import os
import json
import git
import random
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from datasets import load_dataset
import evaluate
from sentence_transformers import SentenceTransformer

# ====================== CONFIG ======================
TEACHER_MODEL = "bert-base-uncased"
STUDENT_BASE = "distilbert-base-uncased"
DATASET_NAME = "glue"
SUBSET = "mnli"
MAX_ITERATIONS = 20
ARCHIVE_DIR = Path("evo_archive")
VALIDATION_THRESHOLD = 0.99  # ≥99% of teacher accuracy required
# =====================================================

# Setup archive (DGM evolutionary tree)
if not ARCHIVE_DIR.exists():
    repo = git.Repo.init(ARCHIVE_DIR)
    (ARCHIVE_DIR / "README.md").write_text("EvoDistill variant archive\n")
    repo.index.add(["README.md"])
    repo.index.commit("init archive")
else:
    repo = git.Repo(ARCHIVE_DIR)

# Load data
raw_datasets = load_dataset(DATASET_NAME, SUBSET)
tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)

def preprocess(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, max_length=128)

encoded = raw_datasets.map(preprocess, batched=True)
train_ds = encoded["train"].shuffle(seed=42).select(range(5000))   # toy size
eval_ds = encoded["validation_matched"].select(range(1000))
