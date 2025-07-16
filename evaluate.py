# evaluate.py
import json
import torch
from utils.metrics import evaluate_scores
from datasets.mimic_dataset import MIMICDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import your model here (adjust path accordingly)
from models.vit_gpt2 import ReportGenerationModel  # change per model

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'checkpoints/vit_gpt2_final.pth'  # change per model
SPLIT_FILE = 'data/splits/test.json'
TOKENISED_DATA = 'data/final_mimic_tokenised.pt'
IMAGE_FOLDER = 'data/cleaned_mimic_image_report_pairs/images/'

def load_model():
    model = ReportGenerationModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

def main():
    print("🔍 Evaluating model on test set...\n")

    dataset = MIMICDataset(split_file=SPLIT_FILE,
                           image_folder=IMAGE_FOLDER,
                           tokenised_data_path=TOKENISED_DATA)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = load_model()

    scores_list = []
    output_json = {}

    for idx, sample in enumerate(tqdm(loader)):
        image = sample['image'].to(DEVICE)
        reference = sample['report'][0]
        sample_id = f"Sample_{idx}"

        with torch.no_grad():
            prediction = model.generate(image)

        prediction_str = prediction[0] if isinstance(prediction, list) else prediction

        scores = evaluate_scores(reference, prediction_str)
        scores_list.append(scores)

        output_json[sample_id] = {
            "real": {
                "FINAL REPORT": reference
            },
            "generated": {
                "FINAL REPORT": prediction_str
            },
            "scores": scores
        }

    # Aggregate Scores
    final_scores = {k: round(sum(score[k] for score in scores_list) / len(scores_list), 3)
                    for k in scores_list[0]}

    print("\n📊 Final Evaluation on Test Set:")
    for metric, value in final_scores.items():
        print(f"- {metric}: {value}")

    with open("output/generated_reports.json", "w") as f:
        json.dump(output_json, f, indent=2)

    print("\n✅ Output saved to output/generated_reports.json")

if __name__ == "__main__":
    main()
