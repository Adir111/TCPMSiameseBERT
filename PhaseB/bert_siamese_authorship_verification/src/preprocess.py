import nltk
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import numpy as np
import random

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')


class Preprocessor:
    def __init__(self, config):
        self.max_length = config['bert']['maximum_sequence_length']
        self.tokenizer = BertTokenizer.from_pretrained(config['bert']['model'])
        self.test_split = config['training']['test_split']

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
            max_length=self.max_length,
            return_tensors='tf'
        )
        return encoded_input

    def __handle_chunk(self, chunk_text, preprocessed_collection):
        tokenized = self.__tokenize_text(chunk_text)
        # input_ids = tokenized['input_ids'].numpy().flatten()
        # attention_mask = tokenized['attention_mask'].numpy().flatten()
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

            if len(tokens) > self.max_length:
                # Split text into chunks of chunk_size words
                num_chunks = len(tokens) // self.max_length
                if len(tokens) % self.max_length != 0:
                    num_chunks += 1

                chunks = [tokens[i * self.max_length: (i + 1) * self.max_length] for i in range(num_chunks)]
                for chunk in chunks:
                    chunk_text = self.tokenizer.convert_tokens_to_string(chunk)
                    tokens_count += self.__handle_chunk(chunk_text, preprocessed_collection)
            else:
                # If no chunking is needed, directly tokenize the text
                tokens_count += self.__handle_chunk(text, preprocessed_collection)
        return preprocessed_collection, tokens_count

    @staticmethod
    def equalize_chunks(chunks_list):
        """
        Helper to balance two lists of chunks to the same length.
        """
        lengths = [len(chunks_list[0]), len(chunks_list[1])]
        i1 = lengths.index(max(lengths))  # index of longer list
        i2 = lengths.index(min(lengths))  # index of shorter list

        repeated_chunks = chunks_list[i2] * (lengths[i1] // lengths[i2])
        repeated_chunks += chunks_list[i2][:lengths[i1] % len(chunks_list[i2])]

        chunks_list[i2] = repeated_chunks

        return chunks_list

    def create_xy(self, impostor_1_chunks, impostor_2_chunks):
        if len(impostor_1_chunks) != len(impostor_2_chunks):
            raise ValueError("Chunk lists must be equal length for pairing.")

        # Prepare each input group
        input_ids_1 = []
        attention_mask_1 = []
        token_type_ids_1 = []

        input_ids_2 = []
        attention_mask_2 = []
        token_type_ids_2 = []

        labels = []

        for c1, c2 in zip(impostor_1_chunks, impostor_2_chunks):
            # Positive pair (same author)
            input_ids_1.append(np.squeeze(c1["input_ids"].numpy(), axis=0))
            attention_mask_1.append(np.squeeze(c1["attention_mask"].numpy(), axis=0))
            token_type_ids_1.append(np.squeeze(c1["token_type_ids"].numpy(), axis=0))

            input_ids_2.append(np.squeeze(c2["input_ids"].numpy(), axis=0))
            attention_mask_2.append(np.squeeze(c2["attention_mask"].numpy(), axis=0))
            token_type_ids_2.append(np.squeeze(c2["token_type_ids"].numpy(), axis=0))

            labels.append(1)

            # Negative pair (different author: swap c1 with a random other c2)
            c2_neg = random.choice(impostor_2_chunks)
            input_ids_1.append(np.squeeze(c1["input_ids"].numpy(), axis=0))
            attention_mask_1.append(np.squeeze(c1["attention_mask"].numpy(), axis=0))
            token_type_ids_1.append(np.squeeze(c1["token_type_ids"].numpy(), axis=0))

            input_ids_2.append(np.squeeze(c2_neg["input_ids"].numpy(), axis=0))
            attention_mask_2.append(np.squeeze(c2_neg["attention_mask"].numpy(), axis=0))
            token_type_ids_2.append(np.squeeze(c2_neg["token_type_ids"].numpy(), axis=0))

            labels.append(0)

        # Convert to arrays
        x = (
            np.stack(input_ids_1),
            np.stack(attention_mask_1),
            np.stack(token_type_ids_1),
            np.stack(input_ids_2),
            np.stack(attention_mask_2),
            np.stack(token_type_ids_2),
        )
        y = np.array(labels, dtype=np.int32)

        (
            input_ids_1_train, input_ids_1_test,
            attention_mask_1_train, attention_mask_1_test,
            token_type_ids_1_train, token_type_ids_1_test,
            input_ids_2_train, input_ids_2_test,
            attention_mask_2_train, attention_mask_2_test,
            token_type_ids_2_train, token_type_ids_2_test,
            y_train, y_test
        ) = train_test_split(
            *x, y, test_size=self.test_split, random_state=42
        )

        x_train = [
            input_ids_1_train,
            attention_mask_1_train,
            token_type_ids_1_train,
            input_ids_2_train,
            attention_mask_2_train,
            token_type_ids_2_train,
        ]

        x_test = [
            input_ids_1_test,
            attention_mask_1_test,
            token_type_ids_1_test,
            input_ids_2_test,
            attention_mask_2_test,
            token_type_ids_2_test,
        ]

        return x_train, y_train, x_test, y_test
