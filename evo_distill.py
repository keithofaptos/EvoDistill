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

# Teacher (oracle)
teacher = AutoModelForSequenceClassification.from_pretrained(TEACHER_MODEL, num_labels=3)
teacher.eval()
teacher.to("cuda" if torch.cuda.is_available() else "cpu")

# Baseline teacher score
metric = evaluate.load("accuracy")
teacher_preds = trainer_predict(teacher, eval_ds)
TEACHER_ACC = metric.compute(predictions=teacher_preds, references=eval_ds["labels"])["accuracy"]
print(f"Teacher baseline accuracy: {TEACHER_ACC:.4f}")

# Helper
def trainer_predict(model, dataset):
    trainer = Trainer(model=model)
    preds = trainer.predict(dataset).predictions
    return np.argmax(preds, axis=1)

# Student class
class EvoStudent:
    def __init__(self, hypers=None):
        self.model = AutoModelForSequenceClassification.from_pretrained(STUDENT_BASE, num_labels=3)
        self.hypers = hypers or {
            "lr": 5e-5,
            "temp": 2.5,
            "alpha": 0.7
        }
        self.score = 0.0
        self.embedding = None

    def train(self):
        # Generate fresh soft labels every time (keeps it honest)
        inputs = tokenizer(train_ds["premise"], train_ds["hypothesis"], truncation=True, padding=True, return_tensors="pt")
        inputs = {k: v.to(teacher.device) for k, v in inputs.items()}
        with torch.no_grad():
            soft_labels = torch.softmax(teacher(**inputs).logits / self.hypers["temp"], dim=-1)

        # Custom loss trainer
        def compute_loss(model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
            kd_loss = torch.nn.KLDivLoss(reduction="batchmean")(
                torch.log_softmax(outputs.logits / self.hypers["temp"], dim=-1),
                soft_labels[:outputs.logits.size(0)].to(outputs.logits.device)
            )
            ce_loss = torch.nn.CrossEntropyLoss()(outputs.logits, labels.to(outputs.logits.device)) if labels is not None else 0
            loss = self.hypers["alpha"] * kd_loss + (1 - self.hypers["alpha"]) * ce_loss
            return (loss, outputs) if return_outputs else loss

        args = TrainingArguments(
            output_dir="./temp", per_device_train_batch_size=32, num_train_epochs=3,
            learning_rate=self.hypers["lr"], evaluation_strategy="no", save_strategy="no",
            report_to=[], disable_tqdm=True
        )
        trainer = Trainer(model=self.model, args=args, train_dataset=train_ds, compute_loss=compute_loss)
        trainer.train()

        # Evaluate
        preds = trainer_predict(self.model, eval_ds)
        self.score = metric.compute(predictions=preds, references=eval_ds["labels"])["accuracy"]
        return self.score

    def embed(self):
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding = embedder.encode(json.dumps(self.hypers))

# =================== MAIN EVOLUTION LOOP ===================
archive = []
base = EvoStudent()
base.train()
base.embed()
archive.append(base)

for it in range(1, MAX_ITERATIONS + 1):
    print(f"\n=== Iteration {it}/{MAX_ITERATIONS} ===")

    # Diversity-weighted parent selection (DGM style)
    embeddings = np.stack([s.embedding for s in archive])
    sims = np.dot(embeddings, embeddings.T)
    novelties = 1 - sims.mean(axis=0)
    weights = np.array([s.score for s in archive]) * novelties
    weights /= weights.sum()
    parent = random.choices(archive, weights=weights, k=1)[0]

    # Simple mutation (you can replace with LLM-generated diffs later)
    child = EvoStudent(hypers={
        "lr": parent.hypers["lr"] * random.uniform(0.5, 2.0),
        "temp": max(1.0, parent.hypers["temp"] + random.uniform(-1, 1)),
        "alpha": np.clip(parent.hypers["alpha"] + random.uniform(-0.2, 0.2), 0.1, 0.9)
    })

    child.train()
    child.embed()

    # Strict no-loss gate
    if child.score >= VALIDATION_THRESHOLD * TEACHER_ACC:
        archive.append(child)
        print(f"New valid child! Score: {child.score:.4f} ({child.score/TEACHER_ACC:.1%} of teacher) Archive size: {len(archive)}")
    else:
        print(f"Rejected (only {child.score/TEACHER_ACC:.1%} of teacher)")

# Save best
best = max(archive, key=lambda x: x.score)
best.model.save_pretrained("best_student")
print(f"\nFinal best: {best.score:.4f} ({best.score/TEACHER_ACC:.2%} of teacher)")
print("Saved to ./best_student — ready for inference!")
