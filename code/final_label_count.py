import json
from pathlib import Path
from collections import Counter

# Set the base directory where all *_final.json files are located
base_dir = Path("../output")  # Change this if needed

label_counter = Counter()
total_sentences = 0

# Recursively find all *_final.json files
json_files = list(base_dir.rglob("*_final.json"))

for json_file in json_files:
    with json_file.open("r", encoding="utf-8") as f:
        labeled_data = json.load(f)

    total_sentences += len(labeled_data)

    for sentence_data in labeled_data.values():
        labels = sentence_data.get("label", [])
        label_counter.update(labels)

# Final output
print("📊 Total Label Counts Across All Files:")
print(f"New   : {label_counter.get('new', 0)}")
print(f"Given : {label_counter.get('given', 0)}")
print(f"None  : {label_counter.get('none', 0)}")
print(f"\n🧾 Total number of sentences: {total_sentences}")
