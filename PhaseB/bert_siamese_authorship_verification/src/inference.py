import torch
from transformers import BertTokenizer
from models.bert_siamese import BertSiameseNetwork


class InferenceEngine:
    def __init__(self, model_path="models/bert_siamese.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertSiameseNetwork()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict_similarity(self, text1, text2):
        inputs = self.tokenizer(text1, text2, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            score = self.model(input_ids, attention_mask, input_ids, attention_mask)

        return score.item()
