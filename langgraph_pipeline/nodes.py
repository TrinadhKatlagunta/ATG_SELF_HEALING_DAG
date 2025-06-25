# langgraph_pipeline/nodes.py

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "./model/model_files/"
CONFIDENCE_THRESHOLD = 0.75  # You can tune this

# Load model and tokenizer once globally
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Get label mapping (emotion dataset)
LABELS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

class InferenceNode:
    @staticmethod
    def run(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        confidence, pred_idx = torch.max(probs, dim=-1)
        label = LABELS[pred_idx.item()]
        confidence_score = confidence.item()
        return {
            "text": text,
            "predicted_label": label,
            "confidence": confidence_score,
            "probs": probs.squeeze().tolist()
        }

class ConfidenceCheckNode:
    @staticmethod
    def run(prediction_output):
        confidence = prediction_output["confidence"]
        if confidence >= CONFIDENCE_THRESHOLD:
            prediction_output["status"] = "accepted"
            return prediction_output
        else:
            prediction_output["status"] = "fallback_needed"
            return prediction_output

class FallbackNode:
    @staticmethod
    def run(prediction_output):
        text = prediction_output["text"]
        predicted = prediction_output["predicted_label"]
        confidence = prediction_output["confidence"]

        print(f"\n[FallbackNode] Low confidence ({confidence*100:.2f}%) for label '{predicted}'")
        print(f"Could you clarify your intent? Was this more like:")
        for i, label in enumerate(LABELS):
            print(f"{i}. {label}")
        user_choice = input("Enter the correct label number (0-5), or press Enter to accept prediction: ").strip()

        if user_choice.isdigit() and 0 <= int(user_choice) < len(LABELS):
            corrected_label = LABELS[int(user_choice)]
            prediction_output["final_label"] = corrected_label
            prediction_output["corrected_by_user"] = True
        else:
            prediction_output["final_label"] = predicted
            prediction_output["corrected_by_user"] = False

        return prediction_output