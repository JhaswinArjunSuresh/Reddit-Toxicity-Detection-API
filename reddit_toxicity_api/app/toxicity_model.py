from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class ToxicityClassifier:
    def __init__(self):
        # For illustration, load pre-finetuned model (many are on Hugging Face)
        self.tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
        self.model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")
        self.model.eval()

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        
        return {
            "non-toxic": float(probs[0]),
            "toxic": float(probs[1])
        }

