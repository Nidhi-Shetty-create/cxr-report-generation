import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class MIMICDataset(Dataset):
    def __init__(self, split_path, image_dir, tokenised_report_path, transform=None):
        with open(split_path, 'r') as f:
            self.samples = json.load(f)

        self.image_dir = image_dir
        self.tokenised_reports = torch.load(tokenised_report_path)
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = os.path.join(self.image_dir, item["image_name"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        report_id = item["report_id"]
        report = self.tokenised_reports[report_id]

        return image, torch.tensor(report)
