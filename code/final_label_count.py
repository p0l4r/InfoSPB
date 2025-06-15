import json
from pathlib import Path
from collections import Counter

# Define the directory containing all *_final.json files
base_dir = Path("../output")  # Change this to your actual output directory path
label_counter = Counter()

# Search for all *_final.json files recursively
json_files = list(base_dir.rglob("*_final.json"))

for json_file in json_files:
    # print(f"Processing file: {json_file}")
    with json_file.open("r", encoding="utf-8") as f:
        labeled_data = json.load(f)

    for sentence_data in labeled_data.values():
        labels = sentence_data.get("label", [])
        label_counter.update(labels)

# Print overall results
print("Total Label Counts Across All Files:")
print(f"New   : {label_counter.get('new', 0)}")
print(f"Given : {label_counter.get('given', 0)}")
print(f"None  : {label_counter.get('none', 0)}")
