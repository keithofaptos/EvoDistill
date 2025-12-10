# EvoDistill — Lossless LLM distillation via Darwin Gödel Machine
# MIT © 2025 Keith of Aptos + Grok
# https://github.com/keithofaptos/EvoDistill

import os, json, git, random, numpy as np, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
from sentence_transformers import SentenceTransformer

TEACHER_MODEL = "bert-base-uncased"
STUDENT_BASE = "distilbert-base-uncased"
DATASET_NAME, SUBSET = "glue", "mnli"
MAX_ITERATIONS = 20
VALIDATION_THRESHOLD = 0.99

raw = load_dataset(DATASET_NAME, SUBSET)
tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
encoded = raw.map(lambda e: tokenizer(e["premise"], e["hypothesis"], truncation=True, max_length=128), batched=True)
train_ds = encoded["train"].shuffle(seed=42).select(range(5000))
eval_ds = encoded["validation_matched"].select(range(1000))

teacher = AutoModelForSequenceClassification.from_pretrained(TEACHER_MODEL, num_labels=3).eval().to("cuda" if torch.cuda.is_available() else "cpu")
metric = evaluate.load("accuracy")

def predict(model, ds):
    preds = Trainer(model=model).predict(ds).predictions
    return np.argmax(preds, axis=1)

TEACHER_ACC = metric.compute(predictions=predict(teacher, eval_ds), references=eval_ds["labels"])["accuracy"]
print(f"Teacher accuracy: {TEACHER_ACC:.4f}")

class EvoStudent:
    def __init__(self, lr=5e-5, temp=2.5, alpha=0.7):
        self.model = AutoModelForSequenceClassification.from_pretrained(STUDENT_BASE, num_labels=3)
        self.lr, self.temp, self.alpha = lr, temp, alpha
        self.score = 0

    def train(self):
        inputs = tokenizer(train_ds["premise"], train_ds["hypothesis"], truncation=True, padding=True, return_tensors="pt").to(teacher.device)
        with torch.no_grad():
            soft = torch.softmax(teacher(**inputs).logits / self.temp, dim=-1)

        def loss_fn(model, inputs, return_outputs=False):
            outputs = model(**{k:v for k,v in inputs.items() if k!="labels"})
            kd = torch.nn.KLDivLoss(reduction="batchmean")(
                torch.log_softmax(outputs.logits/self.temp, dim=-1), soft[:outputs.logits.size(0)])
            ce = torch.nn.CrossEntropyLoss()(outputs.logits, inputs["labels"].to(outputs.logits.device))
            loss = self.alpha * kd + (1-self.alpha) * ce
            return (loss, outputs) if return_outputs else loss

        trainer = Trainer(model=self.model, args=TrainingArguments(output_dir="./temp", per_device_train_batch_size=32,
            num_train_epochs=3, learning_rate=self.lr, report_to=[], disable_tqdm=True),
            train_dataset=train_ds, compute_loss=loss_fn)
        trainer.train()
        self.score = metric.compute(predictions=predict(self.model, eval_ds), references=eval_ds["labels"])["accuracy"]

    def embed(self):
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return embedder.encode(json.dumps({"lr":self.lr,"temp":self.temp,"alpha":self.alpha}))

archive = []
current = EvoStudent()
current.train()
archive.append(current)

for i in range(1, MAX_ITERATIONS+1):
    print(f"\nIteration {i}")
    parent = random.choices(archive, weights=[(s.score * (1-np.mean([np.dot(s.embed(), o.embed()) for o in archive if o!=s]))) for s in archive])[0]
    child = EvoStudent(
        lr=parent.lr * random.uniform(0.5,2),
        temp=max(1.0, parent.temp + random.uniform(-1,1)),
        alpha=np.clip(parent.alpha + random.uniform(-0.2,0.2), 0.1, 0.9))
    child.train()
    if child.score >= VALIDATION_THRESHOLD * TEACHER_ACC:
        archive.append(child)
        print(f"New valid student: {child.score/TEACHER_ACC:.1%} of teacher")
    else:
        print(f"Rejected: {child.score/TEACHER_ACC:.1%}")

best = max(archive, key=lambda x: x.score)
best.model.save_pretrained("best_student")
print(f"\nBest retained {best.score/TEACHER_ACC:.2%} of teacher → saved to ./best_student")
