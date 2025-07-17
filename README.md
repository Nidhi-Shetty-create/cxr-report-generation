#  Chest X-ray Report Generation using Image Captioning and RAG

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
|   ├── rag_model
|   |   ├── generator.py
|   |   ├── image_encoder.py
|   |   ├── rag_pipeline.py
|   |   ├── retriever.py
|   |   ├── text_encoder.py
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
- RAG (ViT+ClinicalBert+FAISS+T5)

---

## 📊 Evaluation Metrics

| 📦 Model                       | 🧪 Val Loss | 🧪 Test Loss | 🔹 BLEU-1 | 🔹 BLEU-4 | 🔹 ROUGE-L | 🔹 METEOR | 🔍 Remarks                            |
| ------------------------------ | ----------: | -----------: | --------: | --------: | ---------: | --------: | ------------------------------------- |
| ViT + GPT-2                    |      1.3667 |       1.2183 |     0.421 |     0.087 |      0.331 |     0.103 | Needs improvement; vague in areas  |
| ResNet-50 + LSTM               |      2.8704 |       2.4091 |     0.297 |     0.045 |      0.229 |     0.071 | Struggles with coherence           |
| DenseNet + Transformer Decoder |      2.6012 |       2.1193 |     0.332 |     0.061 |      0.268 |     0.091 | Slightly better but still redundant |
| Swin Transformer + GPT-2       |      1.2495 |       1.0812 |     0.447 |     0.097 |      0.352 |     0.111 | Better specificity, less repetition |
| ConvNeXt + Transformer Decoder |      1.1063 |       0.9678 |     0.489 |     0.134 |      0.398 |     0.127 | Best model; detailed & structured  |
| RAG                            |      0.8063 |       0.6678 |     0.489 |     0.434 |      0.518 |     0.327 | Best Pipeline                      |

---

## 🚀 How to Train

```bash
# Step 1: Install requirements
pip install -r requirements.txt

# Step 2: Prepare your data (MIMIC-CXR or your dataset)
# Make sure it's in the format expected by `datasets/mimic_dataset.py`

# Step 3: Run training script
python train.py --model <model_name> --epochs <num_epochs> --batch_size <bs> --lr <learning_rate>

# Example:
python train.py --model vit_gpt2 --epochs 20 --batch_size 8 --lr 1e-4
```

## How to Evaluate

```bash
# Evaluate trained model on test set
python evaluate.py --model <model_name> --checkpoint <path_to_checkpoint>

# Example:
python evaluate.py --model convnext_transformer --checkpoint checkpoints/convnext_best.pt
```
---
## 🤝 Acknowledgements

- [MIMIC-CXR dataset](https://physionet.org/content/mimic-cxr/2.0.0/) by PhysioNet   
- [Hugging Face Transformers](https://huggingface.co/transformers)   
- [PyTorch Lightning](https://www.pytorchlightning.ai/) ⚡ *(optional)*  
- Pretrained weights from [Torchvision](https://pytorch.org/vision/stable/index.html) and [timm](https://huggingface.co/docs/timm/index)   
  


