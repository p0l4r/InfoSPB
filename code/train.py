import os
import json
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers.data.data_collator import DataCollatorForTokenClassification 
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from seqeval.metrics import f1_score, precision_score, recall_score

# ==================== CONFIG ====================
DATA_DIR = "../processed_data"  # <-- UPDATE THIS PATH
OUTPUT_DIR = "../output"
MODEL_NAME = "xlm-roberta-base"
SEED = 42
NUM_EPOCHS = 5
TRAIN_RATIO, VAL_RATIO = 0.8, 0.1
# ==================== END CONFIG ====================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------- Load data --------------------
def load_data(data_dir):
    sentences, labels = [], []
    # Walk through all subdirectories
    for subfolder in os.listdir(data_dir):
        subfolder_path = os.path.join(data_dir, subfolder)
        if os.path.isdir(subfolder_path):
            for fname in os.listdir(subfolder_path):
                if fname.endswith("_final.json"):
                    file_path = os.path.join(subfolder_path, fname)
                    print(f"🔄 Loading {file_path}...")
                    with open(file_path, encoding="utf-8") as f:
                        file_data = json.load(f)
                        for entry in file_data.values():
                            sentences.append(entry["sentence"])  # list of tokens
                            labels.append(entry["label"])          # list of labels
    return sentences, labels

# Usage
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "processed_data")
sentences, labels = load_data(DATA_DIR)

# -------------------- Prepare label maps --------------------
unique_labels = sorted({label for seq in labels for label in seq})
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

# -------------------- Tokenizer & Encoding --------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)

def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding=True,
    )
    aligned_labels = []
    for i, label_seq in enumerate(examples["labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = [-100] * len(word_ids)
        for word_idx, label in enumerate(label_seq):
            for token_idx, w_id in enumerate(word_ids):
                if w_id == word_idx:
                    label_ids[token_idx] = label2id[label]
        aligned_labels.append(label_ids)
    tokenized["labels"] = aligned_labels
    return tokenized

# -------------------- Dataset --------------------
dataset = Dataset.from_dict({"tokens": sentences, "labels": labels}).map(
    tokenize_and_align_labels, batched=True
)

# Split into train/val/test
train_size = int(TRAIN_RATIO * len(dataset))
val_size = int(VAL_RATIO * len(dataset))
train_data = dataset.select(range(train_size))
val_data = dataset.select(range(train_size, train_size + val_size))
test_data = dataset.select(range(train_size + val_size, len(dataset)))

# -------------------- Collator & Metrics --------------------
data_collator = DataCollatorForTokenClassification(tokenizer)

def compute_metrics(p):
    preds, labels = p
    preds = np.argmax(preds, axis=2)
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_preds = [
        [id2label[p] for p, l in zip(pred, label) if l != -100] for pred, label in zip(preds, labels)
    ]
    return {
        "f1": f1_score(true_labels, true_preds),
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
    }

# -------------------- Trainer --------------------
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    seed=SEED,
    logging_steps=50,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=tokenizer,
)

# -------------------- Train & Save --------------------
trainer.train()
trainer.evaluate(test_data)

model.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
print("✅ Model and tokenizer saved!")
