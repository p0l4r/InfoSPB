import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset
from infospb import load_data, tokenize_and_align_labels, compute_metrics, count_parameters, evaluate_baseline_performance

# ==================== Load Parameters ====================
with open("./parameter_tuning.json", "r", encoding="utf-8") as f:
    config = json.load(f)

DATA_DIR = config.get("data_dir", "../processed_data")
OUTPUT_DIR = config.get("output_dir", "../output/baseline")
MODEL_NAME = config.get("model_name", "xlm-roberta-base")
SEED = config.get("seed", 42)
BATCH_SIZE = config.get("batch_size", 8)
SPLITS = config.get("splits", {"train": 0.8, "val": 0.10, "test": 0.10})


os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------- Load Data --------------------
sentences, labels = load_data(DATA_DIR)

# -------------------- Label Maps --------------------
unique_labels = sorted({label for seq in labels for label in seq})
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

# -------------------- Tokenizer --------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)

_, test_sents, _, test_labels = train_test_split(
    sentences, labels, test_size=SPLITS["test"], random_state=SEED
)

# Build test dataset
test_dataset = Dataset.from_dict({"tokens": test_sents, "labels": test_labels}).map(
    lambda examples: tokenize_and_align_labels(config=config, examples=examples, label2id=label2id, tokenizer=tokenizer), batched=True
)


# Run baseline evaluation
baseline_results, model, param_stats = evaluate_baseline_performance(
    MODEL_NAME=MODEL_NAME,
    id2label=id2label,
    label2id=label2id,
    tokenizer=tokenizer,
    test_dataset=test_dataset,
    OUTPUT_DIR=OUTPUT_DIR,
    BATCH_SIZE=BATCH_SIZE
)

with open(os.path.join(OUTPUT_DIR, "baseline_results.json"), "w", encoding="utf-8") as f:
    json.dump(baseline_results, f, indent=4)
    
print("\nBaseline results saved to 'baseline_results.json'")
