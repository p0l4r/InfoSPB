import json
from pathlib import Path
from collections import Counter

# Set the base directory where all *_final.json files are located
base_dir = Path("../processed_data")  # Change this if needed

label_counter = Counter()
mention_counter = Counter()
total_sentences = 0

# Recursively find all *_labeled_sentences.json files
json_files = list(base_dir.rglob("*_labeled_sentences.json"))

for json_file in json_files:
    with json_file.open("r", encoding="utf-8") as f:
        labeled_data = json.load(f)

    total_sentences += len(labeled_data)

    for sentence_data in labeled_data.values():
        labels = sentence_data.get("label", [])
        label_counter.update(labels)

# Recursively find all *_mentions.json files to count given/new mentions
mention_json_files = list(base_dir.rglob("*_mentions.json"))

for mention_file in mention_json_files:
    with mention_file.open("r", encoding="utf-8") as f:
        mention_data = json.load(f)

    for mentions_list in mention_data.values():
        for mention in mentions_list:
            mention_counter["total_mentions"] += 1
            mention_type = mention.get("type", "").lower()
            if mention_type == "given":
                mention_counter["given_mentions"] += 1
            elif mention_type == "new":
                mention_counter["new_mentions"] += 1

# Final output
print("Total Label Counts Across All Files:")
print(f"B-new   : {label_counter.get('B-new', 0)}")
print(f"I-new   : {label_counter.get('I-new', 0)}")
print(f"B-given : {label_counter.get('B-given', 0)}")
print(f"I-given : {label_counter.get('I-given', 0)}")
print(f"O       : {label_counter.get('O', 0)}")

print("\nTotal Mention Counts Across All Files:")
print(f"Total Mentions : {mention_counter.get('total_mentions', 0)}")
print(f"Mentions (Given) : {mention_counter.get('given_mentions', 0)}")
print(f"Mentions(New)   : {mention_counter.get('new_mentions', 0)}")