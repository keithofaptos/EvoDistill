from transformers import DataCollatorWithPadding, Trainer, AutoTokenizer, AutoModelForSequenceClassification
import os, json, git, random, numpy as np, torch
from pathlib import Path
from datasets import load_dataset
import evaluate
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

TEACHER_MODEL = "bert-base-uncased"
STUDENT_BASE = "distilbert-base-uncased"
DATASET_NAME, SUBSET = "glue", "mnli"
MAX_ITERATIONS = 20
VALIDATION_THRESHOLD = 0.99

raw = load_dataset(DATASET_NAME, SUBSET)
tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
encoded = raw.map(lambda e: tokenizer(e["premise"], e["hypothesis"], truncation=True, max_length=128, padding="max_length"), batched=True)
train_ds = encoded["train"].shuffle(seed=42).select(range(5000))
eval_ds = encoded["validation_matched"].select(range(1000))
train_ds = train_ds.remove_columns(["premise", "hypothesis"])
eval_ds = eval_ds.remove_columns(["premise", "hypothesis"])

teacher = AutoModelForSequenceClassification.from_pretrained(TEACHER_MODEL, num_labels=3).eval().to("cuda" if torch.cuda.is_available() else "cpu")
metric = evaluate.load("accuracy")

def predict(model, ds):
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(model=model, data_collator=collator)
    return np.argmax(trainer.predict(ds).predictions, axis=1)

TEACHER_ACC = metric.compute(predictions=predict(teacher, eval_ds), references=eval_ds["label"])["accuracy"]
print(f"Teacher accuracy: {TEACHER_ACC:.4f}")

class EvoStudent:
    def __init__(self, lr=5e-5, temp=2.5, alpha=0.7):
        self.model = AutoModelForSequenceClassification.from_pretrained(STUDENT_BASE, num_labels=3)
        self.lr, self.temp, self.alpha = lr, temp, alpha
        self.score = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def train(self):
        teacher_logits_list = []
        batch_size = 8
        print("Generating teacher targets...")
        with torch.no_grad():
            for i in range(0, len(train_ds), batch_size):
                batch_indices = range(i, min(i + batch_size, len(train_ds)))
                batch_subset_with_text = raw["train"].select(batch_indices)
                inputs = tokenizer(list(batch_subset_with_text["premise"]), list(batch_subset_with_text["hypothesis"]), truncation=True, padding=True, return_tensors="pt").to(teacher.device)
                logits = teacher(**inputs).logits
                teacher_logits_list.append(logits.cpu())
        teacher_logits = torch.cat(teacher_logits_list, dim=0)
        soft = torch.softmax(teacher_logits / self.temp, dim=-1).to(self.device)
        collator = DataCollatorWithPadding(tokenizer=tokenizer)
        train_loader = DataLoader(train_ds, batch_size=8, collate_fn=collator, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.model.train()
        idx = 0
        for epoch in range(3):
            for batch in train_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                labels = batch.pop("label", None)
                if labels is None:
                    continue
                outputs = self.model(**batch)
                kd = torch.nn.KLDivLoss(reduction="batchmean")(torch.log_softmax(outputs.logits/self.temp, dim=-1), soft[idx:idx+outputs.logits.size(0)])
                ce = torch.nn.CrossEntropyLoss()(outputs.logits, labels.to(self.device))
                loss = self.alpha * kd + (1-self.alpha) * ce
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                idx += outputs.logits.size(0)
        self.score = metric.compute(predictions=predict(self.model, eval_ds), references=eval_ds["label"])["accuracy"]
        print(f"Student score: {self.score:.4f}")

    def embed(self):
        if not hasattr(self, "_cached_embed"):
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self._cached_embed = embedder.encode(json.dumps({"lr":self.lr,"temp":self.temp,"alpha":self.alpha}))
        return self._cached_embed
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return embedder.encode(json.dumps({"lr":self.lr,"temp":self.temp,"alpha":self.alpha}))

archive = []
current = EvoStudent()
current.train()
archive.append(current)

for i in range(1, MAX_ITERATIONS+1):
    print(f"\nIteration {i}")
    if len(archive) == 1:
        parent = archive[0]
    else:
        weights = []
        for s in archive:
            diversity = np.mean([np.dot(s.embed(), o.embed()) for o in archive if o!=s])
            w = s.score * (1 - diversity)
            weights.append(max(0.01, w))
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        parent = random.choices(archive, weights=weights)[0]
    child = EvoStudent(lr=parent.lr * random.uniform(0.5,2), temp=max(1.0, parent.temp + random.uniform(-1,1)), alpha=np.clip(parent.alpha + random.uniform(-0.2,0.2), 0.1, 0.9))
    child.train()
    if child.score >= VALIDATION_THRESHOLD * TEACHER_ACC:
        archive.append(child)
        print(f"New valid student: {child.score/TEACHER_ACC:.1%} of teacher")
    else:
        print(f"Rejected: {child.score/TEACHER_ACC:.1%}")

best = max(archive, key=lambda x: x.score)
best.model.save_pretrained("best_student")
print(f"\nBest retained {best.score/TEACHER_ACC:.2%} of teacher → saved to ./best_student")
