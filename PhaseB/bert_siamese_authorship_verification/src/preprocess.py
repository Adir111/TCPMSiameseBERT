import re
import nltk
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer
from config.get_config import get_config

# Load config
config = get_config()


class TextPreprocessor:
    def __init__(self, max_length=config['bert']['maximum_sequence_length']):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length
        self.lemmatizer = WordNetLemmatizer()
        try:
            from nltk.corpus import wordnet
            wordnet.ensure_loaded()  # This checks if it can be used
        except LookupError:
            nltk.download('wordnet')
            nltk.download('omw-1.4')

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove punctuation
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words]
        return " ".join(words)

    @staticmethod
    def divide_into_chunk_pair(text_1, text_2, chunk_size=20):
        words_1 = text_1.split()
        words_2 = text_2.split()

        # Find minimum length to avoid index errors
        min_length = min(len(words_1), len(words_2))
        words_1 = words_1[:min_length]
        words_2 = words_2[:min_length]

        # Create chunks
        chunks = []
        for i in range(0, min_length, chunk_size):
            chunk_1 = " ".join(words_1[i:i + chunk_size])
            chunk_2 = " ".join(words_2[i:i + chunk_size])
            chunks.append((chunk_1, chunk_2))
        return chunks

    @staticmethod
    def divide_into_chunk(text, chunk_size=20):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def tokenize_chunk(self, chunk):
        token = self.tokenizer(
            chunk,
            padding="max_length",
            truncation="longest_first",
            max_length=self.max_length,
            return_tensors="pt",
            return_overflowing_tokens=False
        )
        return token["input_ids"], token["attention_mask"]
