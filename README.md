# 🧠 Chest X-Ray Report Generation

A vision-language deep learning project for automatic radiology report generation from chest X-ray images using transformer-based architectures.

![Python](https://img.shields.io/badge/python-3.8+-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Status](https://img.shields.io/badge/status-Active-blue)

---


<details>
<summary><strong>📁 Project Structure</strong> (click to expand)</summary>

```bash
internship/
├── data/                         # ⚠️ Not included in repo (private MIMIC-CXR)
│   ├── cleaned_mimic_image_report_pairs/
│   ├── final_mimic_tokenised.pt
│   └── splits/
│       ├── train.json
│       ├── val.json
│       └── test.json
├── datasets/
│   └── mimic_dataset.py
├── models/
│   ├── vit_gpt2.py
│   ├── resnet_lstm.py
│   ├── densenet_transformer.py
│   ├── swin_gpt2.py
│   └── convnext_transformer.py
├── utils/
│   └── metrics.py
├── prepare_splits.py
├── train.py
├── evaluate.py
└── Data_prep.ipynb
```
</details> 


---

## 🧠 Models Implemented

- ViT (Vision Transformer) + GPT-2
- ResNet-50 + LSTM
- DenseNet + Transformer Decoder
- Swin Transformer + GPT
- ConvNeXt + Transformer Decoder ✅ (Best)

---

## 📊 Evaluation Metrics

| Model                   | BLEU-1 | BLEU-4 | ROUGE-L | METEOR | Report Quality         |
|------------------------|--------|--------|----------|--------|------------------------|
| ViT + GPT-2            | 0.421  | 0.087  | 0.331    | 0.103  | ⚠️ Needs Improvement    |
| Swin Transformer + GPT | 0.341  | 0.069  | 0.262    | 0.093  | 🔁 Redundant            |
| ConvNeXt + Decoder     | 0.489  | 0.134  | 0.398    | 0.127  | ✅ Best Quality          |

---

## 🚀 How to Train

```bash
python train.py --model convnext_transformer

Use the train.py file to specify models, batch sizes, etc.
python evaluate.py --model convnext_transformer

📦 Dependencies
Python 3.8+

PyTorch

Transformers

torchvision

nltk

scikit-learn

bash
Copy
Edit
pip install -r requirements.txt
🛡️ Note
Dataset files like MIMIC-CXR, image-text pairs, and .pt files are not included in the public repo to respect data privacy.

🧾 Credits
MIMIC-CXR Dataset

Hugging Face Transformers

OpenAI GPT-2

📬 Contact
Feel free to connect: Nidhi Shetty
GitHub | LinkedIn

📄 License
MIT License

yaml
Copy
Edit

---

