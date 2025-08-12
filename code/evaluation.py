import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def load_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)

def normalize_mention_tokens(tokens: List[str]) -> str:
    # Normalize mention tokens for matching (lowercase + join)
    return " ".join(tokens).lower().strip()

def mentions_from_ann_dict(ann_dict: Dict) -> List[str]:
    # Extract gold mention texts from ann_dict
    mentions = []
    for ent_info in ann_dict.values():
        # Use normalized annotated_span for matching
        mention_text = ent_info.get("annotated_span", "").lower().strip()
        mentions.append(mention_text)
    return mentions

def mentions_from_mentions_json(mentions_dict: Dict) -> List[Tuple[str, str]]:
    # Extract detected mention texts and their types (given/new)
    mentions = []
    for sent_mentions in mentions_dict.values():
        for mention in sent_mentions:
            norm_text = normalize_mention_tokens(mention['tokens'])
            mention_type = mention.get('type', '').lower()
            mentions.append((norm_text, mention_type))
    return mentions

def evaluate_mentions(gold_mentions: List[str], pred_mentions: List[Tuple[str, str]], gold_ann_dict: Dict, pred_mentions_dict: Dict) -> Dict:
    """
    Evaluate mention detection (based on mention texts) and classification of 'given'/'new'.
    - gold_mentions: list of normalized gold mention texts
    - pred_mentions: list of (normalized mention text, type) tuples
    """
    gold_set = set(gold_mentions)
    pred_set = set([m[0] for m in pred_mentions])

    # Mention detection counts
    true_positives = gold_set.intersection(pred_set)
    false_positives = pred_set.difference(gold_set)
    false_negatives = gold_set.difference(pred_set)

    prec = len(true_positives) / (len(true_positives) + len(false_positives)) if (len(true_positives) + len(false_positives)) > 0 else 0.0
    rec = len(true_positives) / (len(true_positives) + len(false_negatives)) if (len(true_positives) + len(false_negatives)) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # Classification: For mentions found in gold (intersection), check if given/new matches
    # Build mapping from mention text -> label in gold
    gold_label_map = {}
    for ent_id, ent_info in gold_ann_dict.items():
        text = ent_info.get("annotated_span", "").lower().strip()
        label = ent_info.get("label", "").lower()
        gold_label_map[text] = label

    # Build mapping from mention text -> predicted label
    pred_label_map = {}
    for sent_mentions in pred_mentions_dict.values():
        for mention in sent_mentions:
            text = normalize_mention_tokens(mention['tokens'])
            label = mention.get('type', '').lower()
            pred_label_map[text] = label

    # Filter to mentions in true positives only
    common_mentions = true_positives

    y_true = []
    y_pred = []
    for mention_text in common_mentions:
        y_true.append(gold_label_map.get(mention_text, ""))
        y_pred.append(pred_label_map.get(mention_text, ""))

    # Compute classification metrics (only on mentions detected)
    if y_true and y_pred:
        class_prec, class_rec, class_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        class_acc = accuracy_score(y_true, y_pred)
    else:
        class_prec, class_rec, class_f1, class_acc = 0.0, 0.0, 0.0, 0.0

    return {
        "mention_detection": {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "true_positives": len(true_positives),
            "false_positives": len(false_positives),
            "false_negatives": len(false_negatives)
        },
        "classification": {
            "precision": class_prec,
            "recall": class_rec,
            "f1": class_f1,
            "accuracy": class_acc,
            "evaluated_mentions": len(y_true)
        }
    }

if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent.parent
    processed_dir = root_dir / "processed_data"

    all_results = {}

    # Iterate over each page folder in processed_data
    for page_dir in processed_dir.iterdir():
        if not page_dir.is_dir():
            continue

        # We expect files like AAiW_Bengali_001_ann_dict.json etc
        page_name = page_dir.name
        try:
            ann_dict = load_json(next(page_dir.glob("*_ann_dict.json")))
            mentions_dict = load_json(next(page_dir.glob("*_mentions.json")))
            # token_dict and labeled_sentences not needed for evaluation here, but can be loaded if needed
        except Exception as e:
            print(f"Skipping {page_name} due to error loading files: {e}")
            continue

        gold_mentions = mentions_from_ann_dict(ann_dict)
        pred_mentions = mentions_from_mentions_json(mentions_dict)

        eval_result = evaluate_mentions(gold_mentions, pred_mentions, ann_dict, mentions_dict)
        all_results[page_name] = eval_result

    # Save summary
    summary_path = processed_dir / "evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print(f"Evaluation completed. Results saved to {summary_path}")
