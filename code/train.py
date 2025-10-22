import os
import json
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from datasets import Dataset
from transformers import AutoTokenizer
from infospb import load_data, tokenize_and_align_labels, compute_metrics, count_parameters, train_and_evaluate

# ==================== Load Parameters ====================
with open("./parameter_tuning.json", "r", encoding="utf-8") as f:
    config = json.load(f)  # this can be overridden by config.get(...) below if needed for faster experimentation

DATA_DIR = config.get("data_dir", "../processed_data")
OUTPUT_DIR = config.get("output_dir", "../output")
MODEL_NAME = config.get("model_name", "csebuetnlp/banglabert")
SEED = config.get("seed", 42) # don't change this
NUM_EPOCHS = config.get("num_epochs", 50)
BATCH_SIZE = config.get("batch_size", 8)
LEARNING_RATE = config.get("learning_rate", 0.01)
WEIGHT_DECAY = config.get("weight_decay", 0.01)
LR_SCHEDULER_TYPE = config.get("lr_scheduler_type", "linear")
WARMUP_RATIO = config.get("warmup_ratio", 0.1)
SPLITS = config.get("splits", {"train": 0.8, "val": 0.1, "test": 0.1})
K_FOLDS = config.get("k_folds", 10)  # 0 = no CV
FREEZE_BASE_MODEL = config.get("training", {}).get("freeze_base_model", True)

DATA_INTAKE = 30  # This is used to limit data loading for experimentation. This value is equivalent to the number of data files to load.

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------- Load Data --------------------
sentences, labels = load_data(DATA_DIR, data_intake=DATA_INTAKE)

# -------------------- Label Maps --------------------
unique_labels = sorted({label for seq in labels for label in seq})
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

# -------------------- Tokenizer --------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)


if __name__ == "__main__":
    
    if K_FOLDS > 1: # K-Fold Cross-Validation
        print(f"Running {K_FOLDS}-fold cross-validation...")
        kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
        results = []
        for fold, (train_idx, test_idx) in enumerate(kf.split(sentences)):
            train_sents = [sentences[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            test_sents = [sentences[i] for i in test_idx]
            test_labels = [labels[i] for i in test_idx]

            train_sents, val_sents, train_labels, val_labels = train_test_split(
                train_sents, train_labels, test_size=SPLITS["val"], random_state=SEED
            )

            train_dataset = Dataset.from_dict({"tokens": train_sents, "labels": train_labels}).map(
                lambda examples: tokenize_and_align_labels(config=config, examples=examples, label2id=label2id, tokenizer=tokenizer), batched=True
            )
            val_dataset = Dataset.from_dict({"tokens": val_sents, "labels": val_labels}).map(
                lambda examples: tokenize_and_align_labels(config=config, examples=examples, label2id=label2id, tokenizer=tokenizer), batched=True
            )
            test_dataset = Dataset.from_dict({"tokens": test_sents, "labels": test_labels}).map(
                lambda examples: tokenize_and_align_labels(config=config, examples=examples, label2id=label2id, tokenizer=tokenizer), batched=True
            )

            eval_results, model, param_stats = train_and_evaluate(
                MODEL_NAME, id2label, label2id, FREEZE_BASE_MODEL, OUTPUT_DIR, 
                BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, LR_SCHEDULER_TYPE, WARMUP_RATIO, tokenizer, 
                train_dataset, val_dataset, test_dataset, fold=fold
            )
            results.append(eval_results)

        avg_results = {k: np.mean([r[k] for r in results]) for k in results[0].keys()}
        print("Average CV Results:", avg_results)

        with open(os.path.join(OUTPUT_DIR, "best_parameters.json"), "w", encoding="utf-8") as f:
            json.dump({"config": config, "results": avg_results}, f, ensure_ascii=False, indent=2)

    else:  # Standard train/val/test split
        train_sents, temp_sents, train_labels, temp_labels = train_test_split(
            sentences, labels, test_size=(1 - SPLITS["train"]), random_state=SEED
        )
        val_size = SPLITS["val"] / (SPLITS["val"] + SPLITS["test"])
        val_sents, test_sents, val_labels, test_labels = train_test_split(
            temp_sents, temp_labels, test_size=(1 - val_size), random_state=SEED
        )

        train_dataset = Dataset.from_dict({"tokens": train_sents, "labels": train_labels}).map(
            lambda examples: tokenize_and_align_labels(config=config, examples=examples, label2id=label2id, tokenizer=tokenizer), batched=True
        )
        val_dataset = Dataset.from_dict({"tokens": val_sents, "labels": val_labels}).map(
            lambda examples: tokenize_and_align_labels(config=config, examples=examples, label2id=label2id, tokenizer=tokenizer), batched=True
        )
        test_dataset = Dataset.from_dict({"tokens": test_sents, "labels": test_labels}).map(
            lambda examples: tokenize_and_align_labels(config=config, examples=examples, label2id=label2id, tokenizer=tokenizer), batched=True
        )

        eval_results, model, param_stats = train_and_evaluate(
            MODEL_NAME, id2label, label2id, FREEZE_BASE_MODEL, OUTPUT_DIR, 
            BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, LR_SCHEDULER_TYPE, WARMUP_RATIO, tokenizer, 
            train_dataset, val_dataset, test_dataset
        )

        with open(os.path.join(OUTPUT_DIR, "best_parameters.json"), "w", encoding="utf-8") as f:
            json.dump({"config": config, "results": eval_results}, f, ensure_ascii=False, indent=2)

        model.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
        tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
        print("Model and tokenizer saved!")
