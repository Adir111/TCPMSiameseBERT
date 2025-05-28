import nltk
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import numpy as np
import random

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

from PhaseB.bert_siamese_authorship_verification.utilities import make_pairs


class Preprocessor:
    def __init__(self, config, tokenizer=None):
        self.config = config
        self.chunk_size = config['model']['chunk_size']
        self.test_split = config['training']['test_split']
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained(config['bert']['model'])
        else:
            self.tokenizer = tokenizer

    def __tokenize_text(self, text):
        """
        Tokenize text into BERT's format (input_ids, attention_mask).

        Parameters:
        - text (str): Input text

        Returns:
        - dict: Tokenized input with input_ids, attention_mask
        """
        encoded_input = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.chunk_size,
            return_tensors='tf'
        )
        return encoded_input

    def __handle_chunk(self, chunk_text, preprocessed_collection):
        tokenized = self.__tokenize_text(chunk_text)
        tokens_count = len(tokenized['input_ids'])
        preprocessed_collection.append(tokenized)
        return tokens_count

    def preprocess(self, collection):
        """
        Preprocess the collection of text data:
        - Tokenize the text
        - Handle chunking if text is too long
        - Prepare BERT input format

        Parameters:
        - collection (list): List of text data (each is a long text in your case)

        Returns:
        - preprocessed_collection (list): List of dictionaries containing tokenized text data
        """
        preprocessed_collection = []
        tokens_count = 0

        for text in collection:
            # Tokenize the text first
            tokens = self.tokenizer.tokenize(text)

            if len(tokens) > self.chunk_size:
                # Split text into chunks of chunk_size words
                num_chunks = len(tokens) // self.chunk_size
                if len(tokens) % self.chunk_size != 0:
                    num_chunks += 1

                chunks = [tokens[i * self.chunk_size: (i + 1) * self.chunk_size] for i in range(num_chunks)]
                for chunk in chunks:
                    chunk_text = self.tokenizer.convert_tokens_to_string(chunk)
                    tokens_count += self.__handle_chunk(chunk_text, preprocessed_collection)
            else:
                # If no chunking is needed, directly tokenize the text
                tokens_count += self.__handle_chunk(text, preprocessed_collection)
        return preprocessed_collection, tokens_count

    def equalize_chunks(self, chunks_list):
        """
        Helper to balance two lists of chunks to the same length.
        """
        chunk_ratio = self.config['training']['impostor_chunk_ratio']

        lengths = [len(chunks_list[0]), len(chunks_list[1])]
        i1 = lengths.index(max(lengths))  # index of longer list
        i2 = lengths.index(min(lengths))  # index of shorter list

        # Repeat the smaller list enough times
        repeated_chunks = chunks_list[i2] * (lengths[i1] // lengths[i2])

        # Randomly sample the remaining elements needed
        remainder = lengths[i1] - len(repeated_chunks)
        if remainder > 0:
            repeated_chunks += random.sample(chunks_list[i2], remainder)

        chunks_list[i2] = repeated_chunks

        chunks_list[0] *= chunk_ratio
        chunks_list[1] *= chunk_ratio

        return chunks_list

    def create_xy(self, impostor_1_chunks, impostor_2_chunks, pair_fraction=0.1):
        if len(impostor_1_chunks) != len(impostor_2_chunks):
            raise ValueError("Chunk lists must be equal length for pairing.")

        # Prepare each input group
        ids_1, masks_1, types_1 = [], [], []
        ids_2, masks_2, types_2 = [], [], []
        labels = []

        def append_pair(c1, c2, label):
            ids_1.append(np.squeeze(c1["input_ids"].numpy(), axis=0))
            masks_1.append(np.squeeze(c1["attention_mask"].numpy(), axis=0))
            types_1.append(np.squeeze(c1["token_type_ids"].numpy(), axis=0))

            ids_2.append(np.squeeze(c2["input_ids"].numpy(), axis=0))
            masks_2.append(np.squeeze(c2["attention_mask"].numpy(), axis=0))
            types_2.append(np.squeeze(c2["token_type_ids"].numpy(), axis=0))

            labels.append(label)

        def generate_positive_pairs(chunks):
            n = len(chunks)
            if n < 2:
                return
            indices = list(range(n))
            random.shuffle(indices)
            pairs = make_pairs(indices)

            num_pairs_to_take = int(len(pairs) * pair_fraction)
            pairs = pairs[:num_pairs_to_take]

            for pair in pairs:
                c1, c2 = chunks[pair[0]], chunks[pair[1]]
                append_pair(c1, c2, label=1)

            # limit = min(num_pairs or n // 2, n - 1)
            # for i in range(limit):
            #     c1 = chunks[indices[i]]
            #     c2 = chunks[indices[i + 1]]
            #     append_pair(c1, c2, label=1)

        # def generate_negative_pairs(chunks_a, chunks_b, count):
        def generate_negative_pairs(chunks_a, chunks_b):
            n = len(chunks_a)
            if n < 2:
                return
            indices = list(range(n))
            random.shuffle(indices)
            pairs = make_pairs(indices)

            num_pairs_to_take = int(len(pairs) * pair_fraction)
            pairs = pairs[:num_pairs_to_take]

            for pair in pairs:
                c1, c2 = chunks_a[pair[0]], chunks_b[pair[1]]
                append_pair(c1, c2, label=0)
            # for _ in range(count):
            #     c1 = random.choice(chunks_a)
            #     c2 = random.choice(chunks_b)
            #     append_pair(c1, c2, label=0)

        # Balanced generation
        generate_positive_pairs(impostor_1_chunks)
        generate_positive_pairs(impostor_2_chunks)

        # total_positives = len(labels)
        # generate_negative_pairs(impostor_1_chunks, impostor_2_chunks, total_positives)
        generate_negative_pairs(impostor_1_chunks, impostor_2_chunks)

        # Convert to arrays
        x_dict = {
            "input_ids_1": np.stack(ids_1),
            "attention_mask_1": np.stack(masks_1),
            "token_type_ids_1": np.stack(types_1),
            "input_ids_2": np.stack(ids_2),
            "attention_mask_2": np.stack(masks_2),
            "token_type_ids_2": np.stack(types_2),
        }
        y = np.array(labels, dtype=np.int32)

        # Split each field in x_dict independently
        x_train, x_test, y_train, y_test = {}, {}, None, None

        (
            x_train["input_ids_1"], x_test["input_ids_1"],
            x_train["attention_mask_1"], x_test["attention_mask_1"],
            x_train["token_type_ids_1"], x_test["token_type_ids_1"],
            x_train["input_ids_2"], x_test["input_ids_2"],
            x_train["attention_mask_2"], x_test["attention_mask_2"],
            x_train["token_type_ids_2"], x_test["token_type_ids_2"],
            y_train, y_test
        ) = train_test_split(
            x_dict["input_ids_1"], x_dict["attention_mask_1"], x_dict["token_type_ids_1"],
            x_dict["input_ids_2"], x_dict["attention_mask_2"], x_dict["token_type_ids_2"],
            y,
            test_size=self.test_split,
            random_state=42
        )

        y_train = y_train.astype(np.float32).reshape(-1, 1)
        y_test = y_test.astype(np.float32).reshape(-1, 1)

        return x_train, y_train, x_test, y_test
