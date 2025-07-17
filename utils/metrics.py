# utils/metrics.py
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge

def compute_bleu(reference, hypothesis, weights=(1, 0, 0, 0)):
    smoothie = SmoothingFunction().method4
    try:
        return sentence_bleu([reference.split()], hypothesis.split(), weights=weights, smoothing_function=smoothie)
    except ZeroDivisionError:
        return 0.0

def compute_rouge(reference, hypothesis):
    rouge = Rouge()
    try:
        scores = rouge.get_scores(hypothesis, reference)[0]
        return scores["rouge-l"]["f"]
    except:
        return 0.0

def compute_meteor(reference, hypothesis):
    try:
        return meteor_score([reference], hypothesis)
    except:
        return 0.0

def evaluate_scores(reference, prediction):
    return {
        "BLEU-1": round(compute_bleu(reference, prediction, weights=(1, 0, 0, 0)), 3),
        "BLEU-4": round(compute_bleu(reference, prediction, weights=(0.25, 0.25, 0.25, 0.25)), 3),
        "ROUGE-L": round(compute_rouge(reference, prediction), 3),
        "METEOR": round(compute_meteor(reference, prediction), 3),
    }
