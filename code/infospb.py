import os
import json
import numpy as np
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score

def load_data(data_dir, data_intake: int) -> tuple[list[list[str]], list[list[str]]]:
    ''' 
    Load sentences and labels from the specified directory structure.
    Expects subfolders with files named *_labeled_sentences.json.
    Args:
        data_dir: Path to the main data directory.
        data_intake: An integer value to determine how many data files to load. Useful for comparing performance with less data.
    Returns:
        sentences: List of sentences (each sentence is a list of tokens).
        labels: List of label sequences (each label sequence is a list of labels).
    '''
    data_counter = 0
    sentences, labels = [], []
    for subfolder in os.listdir(data_dir):
        subfolder_path = os.path.join(data_dir, subfolder)
        if os.path.isdir(subfolder_path):
            for fname in os.listdir(subfolder_path):
                if fname.endswith("_labeled_sentences.json"):
                    
                    data_counter += 1
                    if data_intake > 0 and data_counter > data_intake: # this is useful for comparing performance with less data
                        return sentences, labels
                    
                    file_path = os.path.join(subfolder_path, fname)
                    print(f"Loading {file_path}...")
                    with open(file_path, encoding="utf-8") as f:
                        file_data = json.load(f)
                        for entry in file_data.values():
                            sentences.append(entry["sentence"])
                            labels.append(entry["label"])
    return sentences, labels


def tokenize_and_align_labels(config, examples, label2id, tokenizer) -> dict:
    ''' 
    Tokenize the input sentences and align the labels with the tokenized output.
    Args:
        config: Configuration dictionary containing parameters like max_length.
        examples: A dictionary with keys "tokens" and "labels".
        label2id: A mapping from label strings to integer IDs.
        tokenizer: A tokenizer instance from Hugging Face's transformers library.
    Returns:
        tokenized: A dictionary containing the tokenized inputs and aligned labels.
    '''
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding=True,
        max_length=config.get("max_length", 128),
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


def compute_metrics(p, id2label) -> dict:
    '''
    Compute metrics for model evaluation.
    Args:
        p: A tuple containing predictions and true labels.
        id2label: A mapping from integer IDs to label strings.
    Returns:
        A dictionary with computed metrics: f1, accuracy, precision, recall.
    '''
    preds, labels = p
    preds = np.argmax(preds, axis=2)
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_preds = [
        [id2label[p] for p, l in zip(pred, label) if l != -100]
        for pred, label in zip(preds, labels)
    ]
    return {
        "f1": f1_score(true_labels, true_preds),
        "accuracy": accuracy_score(true_labels, true_preds),
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
    }

def count_parameters(model) -> tuple[int, int, int]:
    '''
    Count total, trainable, and frozen parameters in the model.
    Args:
        model: A PyTorch model instance.
    Returns:
        A tuple with total, trainable, and frozen parameter counts.
    '''
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen

def train_and_evaluate(
    MODEL_NAME, id2label, label2id, FREEZE_BASE_MODEL, OUTPUT_DIR, 
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, LR_SCHEDULER_TYPE, WARMUP_RATIO, 
    tokenizer, train_dataset, val_dataset, test_dataset, fold=None) -> tuple[dict, object, dict]:
    '''
    Train and evaluate the model.
    Args:
        MODEL_NAME: Name of the pre-trained model.
        id2label: A mapping from integer IDs to label strings.
        label2id: A mapping from label strings to integer IDs.
        FREEZE_BASE_MODEL: Whether to freeze the base model during training.
        OUTPUT_DIR: Directory to save outputs.
        BATCH_SIZE: Batch size for training and evaluation.
        NUM_EPOCHS: Number of training epochs.
        LEARNING_RATE: Learning rate for the optimizer.
        WEIGHT_DECAY: Weight decay for the optimizer.
        LR_SCHEDULER_TYPE: Learning rate scheduler type.
        WARMUP_RATIO: Warmup ratio for learning rate scheduler.
        tokenizer: A tokenizer instance from Hugging Face's transformers library.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        test_dataset: Test dataset.
    Returns:
        A tuple containing evaluation results, the trained model, and parameter statistics.
    '''
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        trust_remote_code=True, # particularly used for BanglaBERT because of huggingface/safetensors issue
        use_safetensors=True # particularly used for BanglaBERT because of huggingface/safetensors issue
    )

    # Freeze backbone if specified
    if FREEZE_BASE_MODEL:
        print("Freezing base model parameters...")
        for param in model.base_model.parameters():
            param.requires_grad = False

    total, trainable, frozen = count_parameters(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters: {frozen:,}")

    args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, f"results_fold{fold}" if fold else "results"),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",
        logging_strategy="steps", 
        logging_steps=50,
        report_to="none",
        fp16=True,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p=p, id2label=id2label),
        tokenizer=tokenizer,
    )

    history = trainer.train()
    eval_results = trainer.evaluate(test_dataset)

    # Extract per-epoch metrics for JSON logging
    training_history = []
    for entry in trainer.state.log_history:
        if 'epoch' in entry:
            epoch_metrics = {
                'epoch': entry.get('epoch'),
                'loss': entry.get('loss'),
                'eval_loss': entry.get('eval_loss'),
                'eval_f1': entry.get('eval_f1'),
                'eval_precision': entry.get('eval_precision'),
                'eval_recall': entry.get('eval_recall')
            }
            training_history.append(epoch_metrics)

    # Plot training curve
    if history:
        train_loss = [m["loss"] for m in trainer.state.log_history if "loss" in m]
        eval_loss = [m["eval_loss"] for m in trainer.state.log_history if "eval_loss" in m]
        print("Eval loss per epoch:", eval_loss)
        plt.figure()
        plt.plot(train_loss, label="train_loss")
        plt.plot(eval_loss, label="eval_loss")
        plt.legend()
        plt.title(f"Training vs Validation Loss (Fold {fold})")
        plt.savefig(os.path.join(OUTPUT_DIR, f"loss_curve_fold{fold}.png"))

    return eval_results, model, {"total": total, "trainable": trainable, "frozen": frozen}



def evaluate_baseline_performance(
    MODEL_NAME, id2label, label2id, tokenizer,
    test_dataset, OUTPUT_DIR, BATCH_SIZE
) -> tuple[dict, object, dict]:
    """
    Evaluate a frozen pretrained model on test data without any fine-tuning.
    Args:
        MODEL_NAME: Name of the pretrained model.
        id2label: Mapping from label IDs to label strings.
        label2id: Mapping from label strings to label IDs.
        tokenizer: A Hugging Face tokenizer instance.
        test_dataset: The test dataset (tokenized and label-aligned).
        OUTPUT_DIR: Directory to save output results.
        BATCH_SIZE: Batch size for evaluation.
    Returns:
        eval_results: Dictionary with precision, recall, f1, accuracy, etc.
        model: Loaded pretrained model.
        param_stats: Dict with total/trainable/frozen parameters.
    """
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    for param in model.parameters():
        param.requires_grad = False

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters: {frozen:,}")

    args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "baseline_results"),
        per_device_eval_batch_size=BATCH_SIZE,
        report_to="none",
        logging_strategy="no",
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p=p, id2label=id2label),
        tokenizer=tokenizer,
    )

    eval_results = trainer.evaluate()
    print("\nBaseline Evaluation Results:")
    for k, v in eval_results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return eval_results, model, {"total": total, "trainable": trainable, "frozen": frozen}
