"""
This module defines the SignalGeneration singleton class, which is responsible for:
- Loading and preprocessing text data (e.g., Shakespeare texts) into tokenized chunks.
- Generating signals by running predictions on the preprocessed texts using trained classifiers.
- Aggregating and visualizing signals for further analysis.
- Saving and loading generated signals from JSON files.

The class depends on external components such as DataLoader for data retrieval,
Preprocessor for tokenization and text chunking, and DataVisualizer for plotting.

The generated signals are typically used in downstream tasks like distance matrix computation,
anomaly detection, and clustering in authorship verification pipelines.
"""

import numpy as np
from pathlib import Path

from .data_loader import DataLoader
from .preprocess import Preprocessor
from PhaseB.bert_siamese_authorship_verification.utilities import DataVisualizer, save_to_json


class SignalGeneration:
    """
    Singleton class responsible for generating and managing signal representations
    for texts processed by Siamese BERT models.

    Loads and preprocesses texts, generates signal arrays from classifier predictions,
    visualizes signals, and saves signals to JSON files.
    """
    _instance = None


    def __new__(cls, config, logger):
        """
        Ensures only one instance of SignalGeneration exists (singleton pattern).

        Args:
            config (dict): Configuration dictionary.
            logger (Logger): Logger instance for logging.

        Returns:
            SignalGeneration: Singleton instance.
        """
        if cls._instance is None:
            cls._instance = super(SignalGeneration, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance


    def __init__(self, config, logger):
        """
        Initializes the SignalGeneration instance.
        Avoids reinitialization if already initialized.

        Args:
            config (dict): Configuration dictionary.
            logger (Logger): Logger instance.
        """
        if self._initialized:
            return  # Avoid reinitializing on repeated instantiations

        self.logger = logger
        self.general_preprocessor = Preprocessor(config)
        self.chunks_per_batch = config['model']['chunk_to_batch_ratio']
        self.data_visualizer = DataVisualizer(config['wandb']['enabled'], logger)
        self.data_loader = DataLoader(config)
        self.data_path = Path(config['data']['organised_data_folder_path'])
        self.signals_folder = config['data']['signals_folder_name']
        self.shakespeare_preprocessed_texts = None

        (self.data_path / self.signals_folder).mkdir(parents=True, exist_ok=True)
        self._initialized = True


    def load_shakespeare_preprocessed_texts(self, reload=False):
        """
        Loads, preprocesses, and equalizes Shakespeare texts for classification.

        This method:
          - Loads all Shakespeare texts.
          - Preprocesses them into chunked token IDs suitable for model input.
          - Tracks the text with the highest number of chunks.
          - Equalizes all other texts to match the longest one using chunk duplication and sampling.
          - Converts chunk data to NumPy arrays for efficient input handling.
          - Caches the result unless `reload=True`.

        Args:
            reload (bool): If True, forces reloading and reprocessing even if data is cached.
        """
        if reload or self.shakespeare_preprocessed_texts is None:
            self.logger.info("Loading shakespeare texts and preprocessing...")
            self.shakespeare_preprocessed_texts = []

            raw_texts = self.data_loader.get_shakespeare_data()
            preprocessed_data = []
            max_chunks = []
            max_name = "NOT FOUND"

            # Step 1: Preprocess texts and track the max-length chunk list
            for text_object in raw_texts:
                text_name = text_object['text_name']
                text = text_object['text']
                self.logger.info(f"Processing text: {text_name}")

                chunks_list, chunks_tokens_count = self.general_preprocessor.preprocess([text])

                chunk_len = len(chunks_list)
                if chunk_len > len(max_chunks):
                    max_chunks = chunks_list
                    max_name = text_name

                preprocessed_data.append({
                    "text_name": text_name,
                    "original_chunks": chunks_list
                })

                self.logger.info(
                    f"Text '{text_name}' processed into {chunk_len} chunks ({chunks_tokens_count} tokens).")

            self.logger.info(f"Maximum number of chunks (Longest text) found: {len(max_chunks)}, text name is {max_name}.")

            # Step 2: Equalize chunks for all texts based on max_chunks
            for item in preprocessed_data:
                text_name = item["text_name"]
                chunks_list = item["original_chunks"]

                if len(chunks_list) != len(max_chunks):
                    self.logger.info(
                        f"Equalizing text '{text_name}' from {len(chunks_list)} to {len(max_chunks)} chunks.")
                    chunks_list = self.general_preprocessor.equalize_chunks([chunks_list, max_chunks], False)[0]

                # Convert chunk dicts to numpy arrays
                text_chunks = {
                    "input_ids": np.stack([c["input_ids"].numpy().squeeze(0) for c in chunks_list]),
                    "attention_mask": np.stack([c["attention_mask"].numpy().squeeze(0) for c in chunks_list]),
                    "token_type_ids": np.stack([c["token_type_ids"].numpy().squeeze(0) for c in chunks_list]),
                }

                self.shakespeare_preprocessed_texts.append({
                    "text_name": text_name,
                    "text_chunks": text_chunks
                })

        self.logger.info(f"Total of {len(self.shakespeare_preprocessed_texts)} shakespeare texts ready for classification.")


    def generate_signals_for_preprocessed_texts(self, classifier, model_name):
        """
        Generates signals from classifier predictions for all preprocessed texts.
        Converts classifier probabilities to binary outputs, aggregates into signal chunks,
        logs and visualizes the signal, then saves signals to disk.

        Args:
            classifier: A trained classifier with a predict method.
            model_name (str): Name used for saving the signal files.
        """
        model_signals = {}
        for text_object in self.shakespeare_preprocessed_texts:
            text_name = text_object['text_name']
            text_chunks = text_object["text_chunks"]
            predictions = np.asarray(classifier.predict({
                "input_ids": text_chunks['input_ids'],
                "attention_mask": text_chunks['attention_mask'],
                "token_type_ids": text_chunks['token_type_ids']
            }))

            binary_outputs = (predictions >= 0.5).astype(int)
            binary_outputs = binary_outputs.flatten().tolist()

            # Aggregate scores into signal chunks
            signal = [
                np.mean(binary_outputs[i:i + self.chunks_per_batch])
                for i in range(0, len(binary_outputs), self.chunks_per_batch)
            ]
            self.logger.log(f"[INFO] Signal representation: {signal}")

            self.logger.info(f"Signal generated for text: {text_name} by model: {model_name}")
            self.data_visualizer.display_signal_plot(signal, text_name, model_name)

            model_signals[text_name] = signal

        self.__save_model_signal(model_name, model_signals)


    def print_all_signals(self):
        """
        Loads and prints all saved model signals from JSON files using the logger.
        Iterates through all saved signal files in the signals folder.
        """
        signals_path = self.data_path / self.signals_folder

        for file in signals_path.glob("*-signals.json"):
            model_name = file.stem.replace("-signals", "")
            model_signals = self.data_loader.get_model_signals(model_name)
            self.logger.log(f"\nModel: {model_name}")

            for text_name, signal in model_signals.items():
                self.logger.log(f"  Text: {text_name}")
                self.logger.log(f"    Signal: {signal}")


    def __get_signal_file_path(self, model_name):
        """
        Constructs the full file path for the signal JSON file of a given model.

        Args:
            model_name (str): The model name used in the file name.

        Returns:
            pathlib.Path: Full path to the signal JSON file.
        """
        file_name = f"{model_name}-signals.json"
        path = self.data_path / self.signals_folder / file_name
        return path


    def signal_already_exists(self, model_name):
        """
        Checks if the signal JSON file for a given model already exists.

        Args:
            model_name (str): The model name to check.

        Returns:
            bool: True if the signal file exists, False otherwise.
        """
        path = self.__get_signal_file_path(model_name)
        return path.exists()


    def __save_model_signal(self, model_name, signal):
        """
        Saves the signal data of a model into a JSON file.

        Args:
            model_name (str): The model name to use in the file name.
            signal (dict): The signal data to save.
        """
        path = self.__get_signal_file_path(model_name)
        save_to_json(signal, path, f"{model_name} Signal data")
