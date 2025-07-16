
import os
import json
import torch
from torch.utils.data import DataLoader
from datasets.mimic_dataset import MIMICDataset
from utils.metrics import compute_metrics
from transformers import AdamW
from tqdm import tqdm
import argparse

# Dynamically import model
def load_model(model_name):
    model_module = __import__(f"models.{model_name}", fromlist=['get_model'])
    return model_module.get_model()

# Save generated reports and scores
def save_outputs(generated, references, model_name):
    output_json = {}
    for i, (gen, ref) in enumerate(zip(generated, references)):
        output_json[f"Sample_{i}"] = {
            "real": ref,
            "generated": gen
        }
    with open(f"generated_{model_name}.json", "w") as f:
        json.dump(output_json, f, indent=2)

    scores = compute_metrics([g["FINAL REPORT"]["FINDINGS"] for g in generated],
                             [r["FINAL REPORT"]["FINDINGS"] for r in references])
    with open(f"scores_{model_name}.json", "w") as f:
        json.dump(scores, f, indent=2)

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        images, reports = batch
        images, reports = images.to(device), reports.to(device)
        loss = model(images, reports)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            images, reports = batch
            images, reports = images.to(device), reports.to(device)
            loss = model(images, reports)
            val_loss += loss.item()
    return val_loss / len(loader)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model_name).to(device)

    train_dataset = MIMICDataset(split='train')
    val_dataset = MIMICDataset(split='val')
    test_dataset = MIMICDataset(split='test')

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=1)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(3):
        print(f"Epoch {epoch+1}/3")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")
        val_loss = validate(model, val_loader, device)
        print(f"Validation Loss: {val_loss:.4f}")

    print("✅ Training Complete. Evaluating on Test Set...")
    generated_outputs = []
    real_outputs = []
    model.eval()
    with torch.no_grad():
        for sample in tqdm(test_dataset, desc="Generating Reports"):
            image, real_report = sample["image"].unsqueeze(0).to(device), sample["report"]
            generated = model.generate(image)
            generated_outputs.append(generated)
            real_outputs.append(real_report)

    save_outputs(generated_outputs, real_outputs, args.model_name)
    print(f"📝 Output saved to generated_{args.model_name}.json and scores_{args.model_name}.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model script name inside models/ (e.g. vit_gpt2, resnet_lstm)")
    args = parser.parse_args()
    main(args)
