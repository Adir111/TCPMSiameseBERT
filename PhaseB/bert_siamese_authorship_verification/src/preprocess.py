import random
import re
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
from config.get_config import get_config

# Load config
config = get_config()


class TextPreprocessor:
    def __init__(self, max_length=config['bert']['maximum_sequence_length']):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length

    @staticmethod
    def clean_text(text):
        # Split text into words, convert to lowercase, remove punctuation from each word, remove non-alphabetic words and filter out stop words
        text = text.lower()
        text = text.replace("\n", " ")  # Replace newlines with spaces
        text = text.replace("\r", " ")  # Replace carriage returns with spaces
        text = text.replace("\t", " ")  # Replace tabs with spaces
        text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space

        words = text.split()
        # Remove non-alphabetic characters
        words = [word for word in words if word.isalpha()]

        # Remove Stopwords
        stop_words = set(nltk.corpus.stopwords.words("english"))
        words = [word for word in words if word not in stop_words]

        # Re-concatenate words into a text
        return " ".join(words)

    @staticmethod
    def create_model_x_y(chunks_imp_1, chunks_imp_2):
        config = get_config()
        chunk_ratio = config['training']['impostor_chunk_ratio']
        chunks = [chunks_imp_1, chunks_imp_2]

        lens = [len(chunks[0]), len(chunks[1])]
        max_idx = lens.index(max(lens))
        min_idx = lens.index(min(lens))

        # Repeat min side to match max
        repeats = lens[max_idx] // lens[min_idx]
        even_sized_chunks = chunks[min_idx] * repeats

        # Add remaining samples if needed (random sample)
        remainder = lens[max_idx] - len(even_sized_chunks)
        even_sized_chunks += random.sample(chunks[min_idx], remainder)

        # Overwrite balanced chunks
        chunks[min_idx] = even_sized_chunks

        chunks[0] *= chunk_ratio
        chunks[1] *= chunk_ratio

        return np.asarray(chunks[0]), np.asarray([0] * len(chunks[0])), np.asarray(chunks[1]), np.asarray([1] * len(chunks[1]))

    def encode_tokenized_chunks(self, tokenized_chunks, max_length):
        input_ids_list = []
        attention_mask_list = []

        for chunk in tokenized_chunks:
            ids = self.tokenizer.convert_tokens_to_ids(chunk[:max_length])
            attention_mask = [1] * len(ids)

            # Pad if shorter than max_length
            padding_length = max_length - len(ids)
            if padding_length > 0:
                ids += [self.tokenizer.pad_token_id] * padding_length
                attention_mask += [0] * padding_length

            input_ids_list.append(ids)
            attention_mask_list.append(attention_mask)

        return {
            "input_ids": tf.convert_to_tensor(input_ids_list),
            "attention_mask": tf.convert_to_tensor(attention_mask_list)
        }

    @staticmethod
    def divide_tokens_into_chunks(tokens, chunk_size):
        tokens = np.asarray(tokens)
        blocks = len(tokens) // chunk_size
        chunks = []
        for i in range(blocks):
            chunks.append(tokens[i * chunk_size:(i + 1) * chunk_size])
        return chunks

    def tokenize_text(self, text):
        tokens = self.tokenizer.tokenize(
            text,
            max_length=self.max_length
        )
        return tokens
