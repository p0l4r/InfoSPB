import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


class TextAnnotationProcessor:
    """
    A class for reading, processing, and labeling annotated text data.
    Suitable for working with both Bengali and English annotation formats.
    """

    def __init__(self, ann_path: Path, text_path: Path):
        self.ann_path = ann_path
        self.text_path = text_path
        self.ann_dict: Dict[str, Any] = {}
        self.token_dict: Dict[int, Dict[str, Any]] = {}
        self.labeled_sentences: Dict[str, Dict[str, List[str]]] = {}

    def read_ann_with_pandas(self) -> pd.DataFrame:
        """
        Reads an .ann annotation file using pandas and returns a DataFrame.
        """
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
        """
        Converts the annotation DataFrame into a dictionary format.
        """
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

    def save_json(self, data: Dict[Any, Any], path: Path) -> None:
        """
        Saves dictionary data to a JSON file at the specified path.
        """
        if any(isinstance(k, int) for k in data.keys()):
            data = {str(k): v for k, v in data.items()}
        with path.open('w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def strip_punctuation(self, text: str) -> str:
        """
        Removes Bengali and English punctuation from the given text.
        """
        punctuation = "।,،:;!?“”‘’\"'()\\[\\]{}—…-।।<>।৷"
        return re.sub(f"[{re.escape(punctuation)}]", " ", text)

    def replace_commas_with_space(self, text: str) -> str:
        """
        Replaces Bengali and English commas with a space.
        """
        return text.replace(',', ' ').replace('،', ' ')

    def read_text_file(self) -> None:
        """
        Reads the annotated text file and tokenizes it, handling punctuation.
        """
        with self.text_path.open('rb') as f:
            text = f.read().decode('utf-8')
            text = self.replace_commas_with_space(text)

        span_char_map = set()
        for ann in self.ann_dict.values():
            span_char_map.update(range(ann["start_pos"], ann["end_pos"]))

        sentences = text.splitlines(keepends=True)

        annotated_spans = sorted(
            (info['annotated_span'] for info in self.ann_dict.values()),
            key=len,
            reverse=True
        )
        escaped_spans = [re.escape(span) for span in annotated_spans if span.strip()]
        pattern = re.compile('|'.join(escaped_spans) + r'|\S+') if escaped_spans else re.compile(r'\S+')

        self.token_dict = {}
        offset = 0
        token_index = 0

        for i, sentence in enumerate(sentences, start=0):
            sent_id = f"S{i}"
            sentence_start = offset

            for match in pattern.finditer(sentence):
                token_text = match.group()
                token_start = sentence_start + match.start()
                token_end = sentence_start + match.end()

                if not any(pos in span_char_map for pos in range(token_start, token_end)):
                    token_text = self.strip_punctuation(token_text).strip()

                if not token_text:
                    continue

                self.token_dict[token_index] = {
                    'token': token_text,
                    'start_pos': token_start,
                    'end_pos': token_end,
                    'sent_id': sent_id
                }
                token_index += 1

            offset += len(sentence)

    def create_labeled_sentences(self) -> None:
        """
        Labels each token in the tokenized text using annotation information.
        """
        annotations = [
            {
                "start_pos": ann["start_pos"],
                "end_pos": ann["end_pos"],
                "label": ann["label"].lower(),
                "matched": False
            }
            for ann in self.ann_dict.values()
        ]
        annotations.sort(key=lambda x: (x["start_pos"], x["end_pos"]))

        sentence_data: Dict[str, Dict[str, List[str]]] = {}

        for idx in sorted(self.token_dict.keys()):
            token_info = self.token_dict[idx]
            token = token_info["token"]
            tok_start = token_info["start_pos"]
            tok_end = token_info["end_pos"]
            sent_id = token_info["sent_id"]

            label = 'none'
            for ann in annotations:
                if ann["matched"]:
                    continue
                if tok_start >= ann["start_pos"] and tok_end <= ann["end_pos"]:
                    label = ann["label"]
                    ann["matched"] = True
                    break

            if sent_id not in sentence_data:
                sentence_data[sent_id] = {"sentence": [], "label": []}

            sentence_data[sent_id]["sentence"].append(token)
            sentence_data[sent_id]["label"].append(label)

        self.labeled_sentences = {
            str(i): sentence_data[sent_key]
            for i, sent_key in enumerate(sorted(sentence_data.keys(), key=lambda x: int(x[1:])), 1)
        }

    def save_selected_labeled_sentences(
        self,
        selected_ids: List[int],
        output_path: Path
    ) -> None:
        """
        Saves a subset of labeled sentences to a readable text file.
        """
        with output_path.open('w', encoding='utf-8') as f:
            for key in sorted(self.labeled_sentences.keys(), key=int):
                if int(key) in selected_ids:
                    sentence = self.labeled_sentences[key]["sentence"]
                    labels = self.labeled_sentences[key]["label"]
                    f.write(
                        f"Sentence: {key}\n"
                        f"token: {sentence}\n"
                        f"label: {labels}\n\n"
                    )


# ------------------------ MAIN ------------------------

if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "data"
    ann_dir = data_dir / "ann"
    txt_dir = data_dir / "txt"
    output_base = root_dir / "output"

    ann_files = list(ann_dir.glob("*.ann"))

    for ann_file in ann_files:
        base_name = ann_file.stem  # filename without extension
        txt_file = txt_dir / f"{base_name}.txt"

        if not txt_file.exists():
            print(f"⚠️ Skipping {base_name}: corresponding .txt file not found.")
            continue

        print(f"🔄 Processing: {base_name}")

        output_dir = output_base / base_name
        output_dir.mkdir(parents=True, exist_ok=True)

        processor = TextAnnotationProcessor(ann_path=ann_file, text_path=txt_file)

        df = processor.read_ann_with_pandas()
        processor.ann_to_dict(df)
        processor.save_json(processor.ann_dict, output_dir / f"{base_name}_ann.json")

        processor.read_text_file()
        processor.save_json(processor.token_dict, output_dir / f"{base_name}_tokens.json")

        processor.create_labeled_sentences()
        processor.save_json(processor.labeled_sentences, output_dir / f"{base_name}_final.json")

        # Sample: save a few selected sentences (selection logic can be customized)
        selected_ids = [1, 2, 3, 4, 5]
        processor.save_selected_labeled_sentences(
            selected_ids,
            output_dir / f"{base_name}_sample.txt"
        )

    print("✅ All files processed.")

