import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


class TextAnnotationProcessor:
    """
    Reads, processes, and labels annotated text data using IOB tagging,
    supporting multi-token mentions and grouping them into spans.
    """

    def __init__(self, ann_path: Path, text_path: Path):
        self.ann_path = ann_path
        self.text_path = text_path
        self.ann_dict: Dict[str, Any] = {}
        self.token_dict: Dict[int, Dict[str, Any]] = {}
        self.labeled_sentences: Dict[str, Dict[str, List[str]]] = {}

    # ---------------------------
    # Reading and preparing annotations
    # ---------------------------

    def read_ann_with_pandas(self) -> pd.DataFrame:
        """Reads .ann annotation file into DataFrame."""
        with self.ann_path.open('r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        data = []
        i = 0
        while i < len(lines) - 1:
            t_line, a_line = lines[i], lines[i + 1]

            if t_line.startswith("T") and a_line.startswith("A"):
                t_parts = t_line.split('\t')
                a_parts = a_line.split()

                entity_id = t_parts[0]
                label_and_pos = t_parts[1].split()
                nominal_type = label_and_pos[0]
                start_pos = int(label_and_pos[1])
                end_pos = int(label_and_pos[2])
                annotated_span = t_parts[2]
                label = a_parts[-1]

                data.append({
                    "entity_id": entity_id,
                    "nominal_type": nominal_type,
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "annotated_span": annotated_span,
                    "label": label,
                })

            i += 2

        return pd.DataFrame(data)

    def ann_to_dict(self, df: pd.DataFrame) -> None:
        """Stores annotation info in dictionary form."""
        self.ann_dict = {
            row['entity_id']: {
                'start_pos': row['start_pos'],
                'end_pos': row['end_pos'],
                'nominal_type': row['nominal_type'],
                'annotated_span': row['annotated_span'],
                'label': row['label']
            }
            for _, row in df.iterrows()
        }

    # ---------------------------
    # Tokenization and labeling
    # ---------------------------

    def strip_punctuation(self, text: str) -> str:
        """Removes Bengali and English punctuation except hyphen (split separately)."""
        punctuation = "।,،:;!?“”‘’\"'()\\[\\]{}—…।।<>।৷"
        return re.sub(f"[{re.escape(punctuation)}]", " ", text)

    def replace_commas_with_space(self, text: str) -> str:
        """Replace commas with space."""
        return text.replace(',', ' ').replace('،', ' ')

    def read_text_file(self) -> None:
        """
        Tokenizes text word-by-word, splitting on spaces and hyphens.
        Keeps start/end char offsets for alignment with annotations.
        """
        with self.text_path.open('rb') as f:
            text = f.read().decode('utf-8')
            text = self.replace_commas_with_space(text)

        sentences = text.splitlines(keepends=True)

        # Match sequences of non-space, then split hyphen if present
        token_pattern = re.compile(r'\S+')

        self.token_dict = {}
        offset = 0
        token_index = 0

        for i, sentence in enumerate(sentences):
            sent_id = f"S{i}"
            sentence_start = offset

            for match in token_pattern.finditer(sentence):
                raw_token = match.group()
                tok_start = sentence_start + match.start()
                tok_end = sentence_start + match.end()

                # Split on hyphen but keep position mapping
                subtokens = raw_token.split('-')
                running_pos = tok_start
                for j, sub in enumerate(subtokens):
                    clean_token = self.strip_punctuation(sub).strip()
                    if not clean_token:
                        running_pos += len(sub) + (1 if j < len(subtokens) - 1 else 0)
                        continue

                    self.token_dict[token_index] = {
                        'token': clean_token,
                        'start_pos': running_pos,
                        'end_pos': running_pos + len(sub),
                        'sent_id': sent_id
                    }
                    token_index += 1
                    running_pos += len(sub) + (1 if j < len(subtokens) - 1 else 0)

            offset += len(sentence)

    def create_labeled_sentences(self) -> None:
        """Applies IOB tagging scheme with 'new'/'given' labels."""
        annotations = [
            {
                "start_pos": ann["start_pos"],
                "end_pos": ann["end_pos"],
                "label": ann["label"].lower(),
            }
            for ann in self.ann_dict.values()
        ]
        annotations.sort(key=lambda x: (x["start_pos"], x["end_pos"]))

        token_labels = ['O'] * len(self.token_dict)

        for ann in annotations:
            ann_start = ann["start_pos"]
            ann_end = ann["end_pos"]
            label = ann["label"]

            tokens_in_span = []
            for idx, token_info in self.token_dict.items():
                tok_start = token_info["start_pos"]
                tok_end = token_info["end_pos"]

                if not (tok_end <= ann_start or tok_start >= ann_end):
                    tokens_in_span.append(idx)

            if tokens_in_span:
                token_labels[tokens_in_span[0]] = f'B-{label}'
                for idx in tokens_in_span[1:]:
                    token_labels[idx] = f'I-{label}'

        # Group by sentence
        sentence_data: Dict[str, Dict[str, List[str]]] = {}
        for idx, token_info in self.token_dict.items():
            sent_id = token_info["sent_id"]
            if sent_id not in sentence_data:
                sentence_data[sent_id] = {"sentence": [], "label": []}
            sentence_data[sent_id]["sentence"].append(token_info["token"])
            sentence_data[sent_id]["label"].append(token_labels[idx])

        self.labeled_sentences = {
            str(i): sentence_data[sent_key]
            for i, sent_key in enumerate(sorted(sentence_data.keys(), key=lambda x: int(x[1:])), 1)
        }

    # ---------------------------
    # Mention grouping
    # ---------------------------

    def group_mentions(self) -> Dict[str, List[Dict[str, Any]]]:
        """Groups contiguous B-/I- sequences into mention spans for each sentence."""
        mention_dict: Dict[str, List[Dict[str, Any]]] = {}

        for sent_id, data in self.labeled_sentences.items():
            tokens = data["sentence"]
            labels = data["label"]
            mentions = []

            i = 0
            while i < len(tokens):
                label = labels[i]
                if label.startswith("B-"):
                    mention_tokens = [tokens[i]]
                    mention_type = label[2:]
                    start_index = i
                    i += 1
                    while i < len(tokens) and labels[i] == f"I-{mention_type}":
                        mention_tokens.append(tokens[i])
                        i += 1
                    mentions.append({
                        "type": mention_type,
                        "tokens": mention_tokens,
                        "start_index": start_index,
                        "end_index": i - 1
                    })
                else:
                    i += 1

            mention_dict[sent_id] = mentions

        return mention_dict

    # ---------------------------
    # Mention linking (no linking, only global IDs)
    # ---------------------------

    def link_given_mentions(self, mention_dict: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Assigns global mention IDs without any entity linking or entity_id assignment.
        """
        global_counter = 1

        for sent_id in sorted(mention_dict.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0):
            for mention in mention_dict[sent_id]:
                mention['global_mention_id'] = f"M{global_counter}"
                global_counter += 1

        return mention_dict

    # ---------------------------
    # Utility
    # ---------------------------

    def save_json(self, data: Dict[Any, Any], path: Path) -> None:
        if any(isinstance(k, int) for k in data.keys()):
            data = {str(k): v for k, v in data.items()}
        with path.open('w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


# ------------------------ MAIN ------------------------

if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "raw_data"
    ann_dir = data_dir / "ann"
    txt_dir = data_dir / "txt"
    output_base = root_dir / "processed_data"

    ann_files = list(ann_dir.glob("*.ann"))

    for ann_file in ann_files:
        base_name = ann_file.stem
        txt_file = txt_dir / f"{base_name}.txt"

        if not txt_file.exists():
            print(f"⚠️ Skipping {base_name}: corresponding .txt file not found.")
            continue

        print(f"🔄 Processing: {base_name}")

        output_dir = output_base / base_name
        output_dir.mkdir(parents=True, exist_ok=True)

        processor = TextAnnotationProcessor(ann_file, txt_file)

        df = processor.read_ann_with_pandas()
        processor.ann_to_dict(df)

        processor.read_text_file()
        processor.create_labeled_sentences()

        mention_spans = processor.group_mentions()
        linked_mentions = processor.link_given_mentions(mention_spans)

        processor.save_json(processor.ann_dict, output_dir / f"{base_name}_ann_dict.json")
        processor.save_json(processor.token_dict, output_dir / f"{base_name}_token_dict.json")
        processor.save_json(processor.labeled_sentences, output_dir / f"{base_name}_labeled_sentences.json")
        processor.save_json(linked_mentions, output_dir / f"{base_name}_mentions.json")

        print(f"✅ Finished processing: {base_name}\n")
