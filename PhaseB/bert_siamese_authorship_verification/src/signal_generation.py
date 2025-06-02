import numpy as np
from pathlib import Path

from .data_loader import DataLoader
from .preprocess import Preprocessor
from PhaseB.bert_siamese_authorship_verification.utilities import DataVisualizer, save_to_json


class SignalGeneration:
    _instance = None


    def __new__(cls, config, logger):
        if cls._instance is None:
            cls._instance = super(SignalGeneration, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance


    def __init__(self, config, logger):
        if self._initialized:
            return  # Avoid reinitializing on repeated instantiations

        self.logger = logger
        self.general_preprocessor = Preprocessor(config)
        self.chunks_per_batch = config['model']['chunk_to_batch_ratio']
        self.data_visualizer = DataVisualizer(config['wandb']['enabled'], logger)
        self.data_loader = DataLoader(config)
        self.all_signals = {}
        self.signals_path = Path(config['data']['organised_data_folder_path']) / config['data']['signals']

        self._initialized = True


    def generate_signals_for_text(self, shakespearian_text, trained_network):
        text_name = shakespearian_text['text_name']

        chunks_list, chunks_tokens_count = self.general_preprocessor.preprocess([shakespearian_text['text']])
        text_chunks = {
            "input_ids": np.stack([c["input_ids"].numpy().squeeze(0) for c in chunks_list]),
            "attention_mask": np.stack([c["attention_mask"].numpy().squeeze(0) for c in chunks_list]),
            "token_type_ids": np.stack([c["token_type_ids"].numpy().squeeze(0) for c in chunks_list]),
        }

        self.logger.info(
            f"Text '{text_name}' has been preprocessed into {len(chunks_list)} chunks with {chunks_tokens_count} tokens.")

        model_name, model_creator = next(iter(trained_network.items()))
        classifier = model_creator.get_encoder_classifier()
        self.logger.info(f"Generating signal from model: {model_name}...")

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

        # Assign signal to the current text under this model
        if model_name not in self.all_signals:
            self.all_signals[model_name] = {}
        self.all_signals[model_name][text_name] = signal

        self.logger.info(f"Signal generated for text: {text_name} by model: {model_name}")
        self.data_visualizer.display_signal_plot(signal, text_name, model_name)


    def print_all_signals(self):
        for model_name, texts in self.all_signals.items():
            self.logger.log(f"\nModel: {model_name}")
            for text_name, signal in texts.items():
                self.logger.log(f"  Text: {text_name}")
                self.logger.log(f"    Signal: {signal}")


    def save_all_signals(self):
        """
        Save all generated signals to a JSON file.

        Parameters:
        - output_dir: Directory where the file should be saved (str or Path)
        - filename: Name of the output JSON file (default: 'signals.json')
        """
        save_to_json(self.all_signals, self.signals_path, "Signal data")
