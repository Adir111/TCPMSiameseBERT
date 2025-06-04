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

        self.config = config
        self.logger = logger
        self.general_preprocessor = Preprocessor(config)
        self.chunks_per_batch = config['model']['chunk_to_batch_ratio']
        self.data_visualizer = DataVisualizer(config['wandb']['enabled'], logger)
        self.data_loader = DataLoader(config)
        self.data_path = Path(config['data']['organised_data_folder_path'])
        self.signals_folder = config['data']['signals_folder_path']
        self.all_signals = {}
        self.shakespeare_preprocessed_texts = None

        self._initialized = True


    def load_shakespeare_preprocessed_texts(self, reload=False):
        if reload or self.shakespeare_preprocessed_texts is None:
            self.logger.info("Loading shakespeare preprocessed texts...")
            self.shakespeare_preprocessed_texts = []
            tested_collection_texts = self.data_loader.get_shakespeare_data()
            for text_object in tested_collection_texts:
                text_name = text_object['text_name']
                text = text_object['text']
                self.logger.info(f"Processing text: {text_name}")
                chunks_list, chunks_tokens_count = self.general_preprocessor.preprocess(text)
                text_chunks = {
                    "input_ids": np.stack([c["input_ids"].numpy().squeeze(0) for c in chunks_list]),
                    "attention_mask": np.stack([c["attention_mask"].numpy().squeeze(0) for c in chunks_list]),
                    "token_type_ids": np.stack([c["token_type_ids"].numpy().squeeze(0) for c in chunks_list]),
                }
                self.logger.info(f"Text '{text_name}' has been preprocessed into {len(chunks_list)} chunks with {chunks_tokens_count} tokens.")
                text_object = {
                    "text_name": text_name,
                    "text_chunks": text_chunks
                }
                self.shakespeare_preprocessed_texts.append(text_object)

        else:
            self.logger.warn(f"Shakespeare preprocessed texts already loaded.")

        self.logger.info(f"Total of {len(self.shakespeare_preprocessed_texts)} shakespeare texts ready for classification.")


    def generate_signals_for_preprocessed_texts(self, classifier, model_name):
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

            # Assign signal to the current text under this model
            if model_name not in self.all_signals:
                self.all_signals[model_name] = {}
            self.all_signals[model_name][text_name] = signal

            self.logger.info(f"Signal generated for text: {text_name} by model: {model_name}")
            self.data_visualizer.display_signal_plot(signal, text_name, model_name)


    def print_all_signals(self):
        """
        Load all signals from JSON and print them using logger.
        """
        all_signals = self.data_loader.get_all_signals()
        for model_name, texts in all_signals.items():
            self.logger.log(f"\nModel: {model_name}")
            for text_name, signal in texts.items():
                self.logger.log(f"  Text: {text_name}")
                self.logger.log(f"    Signal: {signal}")


    def save_model_signal(self, model_name):
        """
        Saves given model signal into a file
        """
        model_signals = self.all_signals[model_name]

        file_name = f"{model_name}-signals.json"
        path = self.data_path / self.signals_folder / file_name
        save_to_json(model_signals, path, f"{model_name} Signal data")

        # Also update the full all_signals JSON file after saving this model's signal
        all_signals_path = self.data_path / self.config['data']['all_signals']
        all_signals = self.data_loader.get_all_signals()
        all_signals[model_name] = model_signals
        save_to_json(all_signals, all_signals_path, "Updated all_signals with latest model signal")
