import nltk
from transformers import BertTokenizer

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

    def preprocess(self, collection, chunk_size):
        """
        Preprocess the collection of text data:
        - Tokenize the text
        - Handle chunking if text is too long
        - Prepare BERT input format

        Parameters:
        - collection (list): List of text data (each is a long text in your case)
        - chunk_size (int): Size of each chunk if text exceeds max_length (default=None)

        Returns:
        - preprocessed_collection (list): List of dictionaries containing tokenized text data
        """
        preprocessed_collection = []
        tokens_count = 0

        for text in collection:
            # Check if chunk_size is provided and the text is too long for BERT
            if chunk_size is not None and len(text.split()) > self.max_length:
                # Split text into chunks of chunk_size words
                num_chunks = len(text.split()) // chunk_size
                if len(text.split()) % chunk_size != 0:
                    num_chunks += 1

                chunks = [text.split()[i * chunk_size: (i + 1) * chunk_size] for i in range(num_chunks)]
                # Join the chunks back into text and tokenize each chunk
                for chunk in chunks:
                    chunk_text = ' '.join(chunk)
                    tokenized = self.tokenize_text(chunk_text)
                    tokens_count += len(tokenized['input_ids'].numpy().flatten())
                    preprocessed_collection.append(tokenized)
            else:
                # If no chunking is needed, directly tokenize the text
                tokenized = self.tokenize_text(text)
                preprocessed_collection.append(tokenized)
        return preprocessed_collection, tokens_count
