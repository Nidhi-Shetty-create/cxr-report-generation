import os
import json
import random
from tqdm import tqdm

# Set seed for reproducibility
random.seed(42)

# Constants
DATA_DIR = os.path.join("data", "cleaned_mimic_image_report_pairs")
SPLIT_DIR = os.path.join("data", "splits")

# Split percentages
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Create output split directory if not exists
os.makedirs(SPLIT_DIR, exist_ok=True)

def is_valid_pair(report_text):
    return report_text and len(report_text.strip()) > 10

def get_all_pairs():
    pairs = []
    for filename in tqdm(os.listdir(DATA_DIR), desc="Scanning data"):
        if filename.endswith(".json"):
            image_filename = filename.replace(".json", ".png")
            image_path = os.path.join(DATA_DIR, image_filename)
            report_path = os.path.join(DATA_DIR, filename)

            if not os.path.exists(image_path):
                continue  # Skip if image doesn't exist

            with open(report_path, "r") as f:
                report_data = json.load(f)

            # Extract main report text (adapt as needed)
            if isinstance(report_data, dict):
                report_text = report_data.get("report", "")
            else:
                report_text = report_data

            if is_valid_pair(report_text):
                pairs.append({
                    "id": filename.replace(".json", ""),
                    "image_path": os.path.join("cleaned_mimic_image_report_pairs", image_filename),
                    "report": report_text.strip()
                })

    return pairs

def split_and_save(pairs):
    random.shuffle(pairs)
    total = len(pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_split = pairs[:train_end]
    val_split = pairs[train_end:val_end]
    test_split = pairs[val_end:]

    print(f"Total: {total} | Train: {len(train_split)} | Val: {len(val_split)} | Test: {len(test_split)}")

    with open(os.path.join(SPLIT_DIR, "train.json"), "w") as f:
        json.dump(train_split, f, indent=2)
    with open(os.path.join(SPLIT_DIR, "val.json"), "w") as f:
        json.dump(val_split, f, indent=2)
    with open(os.path.join(SPLIT_DIR, "test.json"), "w") as f:
        json.dump(test_split, f, indent=2)

if __name__ == "__main__":
    all_pairs = get_all_pairs()
    split_and_save(all_pairs)
