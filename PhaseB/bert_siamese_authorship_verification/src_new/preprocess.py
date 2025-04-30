import nltk
from transformers import BertTokenizer
import tensorflow as tf

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')


class Preprocessor:
    def __init__(self, config):
        self._config = config
        self.max_length = config['bert']['maximum_sequence_length']
        self.tokenizer = BertTokenizer.from_pretrained(config['bert']['model'])

    def tokenize_text(self, text):
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
                    tokenized = self.tokenize_text(chunk_text)
                    tokens_count += len(tokenized['input_ids'].numpy().flatten())
                    preprocessed_collection.append({
                        'input_ids': tokenized['input_ids'][0],
                        'attention_mask': tokenized['attention_mask'][0]
                    })
            else:
                # If no chunking is needed, directly tokenize the text
                tokenized = self.tokenize_text(text)
                preprocessed_collection.append(tokenized)
        return preprocessed_collection, tokens_count

    @staticmethod
    def equalize_chunks(chunks_list):
        """
        Helper to balance two lists of chunks to the same length.
        """
        lengths = [len(chunks_list[0]), len(chunks_list[1])]
        i1 = lengths.index(max(lengths)) # index of longer list
        i2 = lengths.index(min(lengths)) # index of shorter list

        repeated_chunks = chunks_list[i2] * (lengths[i1] // lengths[i2])
        repeated_chunks += chunks_list[i2][:lengths[i1] % len(chunks_list[i2])]

        chunks_list[i2] = repeated_chunks

        return chunks_list

    @staticmethod
    def create_xy(impostor1, impostor2):
        # Unpack the datasets
        impostor1_batches, impostor2_batches = impostor1[0], impostor2[0]

        # Map impostor1 to label 1, impostor2 to label 2
        def map_with_label(label):
            def wrapper(example):
                return ({
                            'input_text_1': example['input_ids'],
                            'attention_mask_1': example['attention_mask'],
                            'input_text_2': example['input_ids'],
                            'attention_mask_2': example['attention_mask'],
                        }, label)

            return wrapper

        labeled_impostor1 = impostor1_batches.map(map_with_label(tf.constant(1, dtype=tf.int32)))
        labeled_impostor2 = impostor2_batches.map(map_with_label(tf.constant(2, dtype=tf.int32)))

        # Concatenate the datasets
        dataset = labeled_impostor1.concatenate(labeled_impostor2)
        dataset = dataset.shuffle(buffer_size=1000)
        return dataset
