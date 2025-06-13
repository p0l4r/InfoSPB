import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any
import re


def read_ann_with_pandas(ann_file_path: Path) -> pd.DataFrame:
    with ann_file_path.open('r', encoding='utf-8') as f:
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
            label = a_parts[-1]  # last token in the attribute line

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


def ann_to_dict(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        row['entity_id']: {
            'start_pos': row['start_pos'],
            'end_pos': row['end_pos'],
            'nominal_type': row['nominal_type'],
            'annotated_span': row['annotated_span'],
            'label': row['label']
        }
        for _, row in df.iterrows()
    }


def save_json(data: Dict[Any, Any], json_path: Path):
    if any(isinstance(k, int) for k in data.keys()):
        data = {str(k): v for k, v in data.items()}
    with json_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_text_file(text_file_path: Path, ann_dict: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    with open(text_file_path, 'rb') as f:
        text = f.read().decode('utf-8')  # preserve \r\n as-is

    sentences = text.splitlines(keepends=True)  # keep line endings to count them properly

    annotated_spans = sorted(
        (info['annotated_span'] for info in ann_dict.values()),
        key=len,
        reverse=True
    )
    escaped_spans = [re.escape(span) for span in annotated_spans if span.strip()]
    pattern = re.compile('|'.join(escaped_spans) + r'|\S+') if escaped_spans else re.compile(r'\S+')

    token_dict = {}
    offset = 0
    token_index = 0

    for i, sentence in enumerate(sentences, 1):
        sent_id = f"S{i}"
        sentence_start = offset

        for match in pattern.finditer(sentence):
            token_text = match.group()
            token_start = sentence_start + match.start()
            token_end = sentence_start + match.end()

            token_dict[token_index] = {
                'token': token_text,
                'start_pos': token_start,
                'end_pos': token_end,
                'sent_id': sent_id
            }
            token_index += 1

        offset += len(sentence)  # include \r\n

    return token_dict

def create_labeled_sentences(ann_dict: Dict[str, Any], token_dict: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    annotations = [
        {
            "start_pos": ann["start_pos"],
            "end_pos": ann["end_pos"],
            "label": ann["label"].lower(),
            "matched": False
        }
        for ann in ann_dict.values()
    ]

    annotations.sort(key=lambda x: (x["start_pos"], x["end_pos"]))

    sentence_data = {}
    for idx in sorted(token_dict.keys()):
        token_info = token_dict[idx]
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

    final_result = {}
    for idx, sent_key in enumerate(sorted(sentence_data.keys(), key=lambda x: int(x[1:])), 1):
        final_result[str(idx)] = sentence_data[sent_key]

    return final_result


from pathlib import Path
from typing import Dict, List, Any


def save_selected_labeled_sentences(
    final_labeled: Dict[str, Dict[str, List[str]]],
    selected_sentence_ids: List[int],
    output_path: Path
) -> None:
    """
    Save selected labeled sentences to a text file in readable format.

    Parameters:
    - final_labeled: Dictionary of labeled sentences, structured as {sentence_id: {"sentence": [...], "label": [...]}}
    - selected_sentence_ids: List of sentence numbers (as integers) to save
    - output_path: Path to the output .txt file
    """
    with output_path.open('w', encoding='utf-8') as f:
        for key in sorted(final_labeled.keys(), key=int):
            if int(key) in selected_sentence_ids:
                sentence = final_labeled[key]["sentence"]
                labels = final_labeled[key]["label"]

                output = (
                    f"Sentence: {key}\n"
                    f"token: {sentence}\n"
                    f"label: {labels}\n\n"
                )
                f.write(output)


if __name__ == "__main__":
    ann_path = Path(r"D:\0. A_MSc_Thesis\Alice in wonderland\AAiW\InfoSPB\AAiW_Bengali_001.ann")
    text_path = Path(r"D:\0. A_MSc_Thesis\Alice in wonderland\AAiW\InfoSPB\AAiW_Bengali_001.txt")
    json_path_ann = Path(r"D:\0. A_MSc_Thesis\Alice in wonderland\AAiW\InfoSPB\code\AAiW_Bengali_001_ann.json")
    json_path_text_tokens = Path(r"D:\0. A_MSc_Thesis\Alice in wonderland\AAiW\InfoSPB\code\AAiW_Bengali_001_text_tokens.json")
    json_path_final = Path(r"D:\0. A_MSc_Thesis\Alice in wonderland\AAiW\InfoSPB\code\AAiW_Bengali_001_final.json")

    df = read_ann_with_pandas(ann_path)
    result_dict_ann = ann_to_dict(df)
    save_json(result_dict_ann, json_path_ann)

    token_dict = read_text_file(text_path, result_dict_ann)
    save_json(token_dict, json_path_text_tokens)

    final_labeled = create_labeled_sentences(result_dict_ann, token_dict)
    save_json(final_labeled, json_path_final)
    
    output_path = Path(r"D:\0. A_MSc_Thesis\Alice in wonderland\AAiW\InfoSPB\code\sample_labeled_sentences.txt")
    selected_ids = [3, 13, 16, 22]

    save_selected_labeled_sentences(final_labeled, selected_ids, output_path)

