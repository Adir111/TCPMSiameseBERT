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
        self.data_path = Path(config['data']['organised_data_folder_path'])
        self.signals_folder = config['data']['signals_folder_name']
        self.shakespeare_preprocessed_texts = None

        (self.data_path / self.signals_folder).mkdir(parents=True, exist_ok=True)
        self._initialized = True


    def load_shakespeare_preprocessed_texts(self, reload=False):
        if reload or self.shakespeare_preprocessed_texts is None:
            self.logger.info("Loading shakespeare texts and preprocessing...")
            self.shakespeare_preprocessed_texts = []
            tested_collection_texts = self.data_loader.get_shakespeare_data()
            for text_object in tested_collection_texts:
                text_name = text_object['text_name']
                text = text_object['text']
                self.logger.info(f"Processing text: {text_name}")
                chunks_list, chunks_tokens_count = self.general_preprocessor.preprocess([text])
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
        Load each model's signals from its own JSON file and print them using logger.
        """
        signals_path = self.data_path / self.signals_folder

        for file in signals_path.glob("*-signals.json"):
            model_name = file.stem.replace("-signals", "")
            model_signals = self.data_loader.get_model_signals(model_name)
            self.logger.log(f"\nModel: {model_name}")

            for text_name, signal in model_signals.items():
                self.logger.log(f"  Text: {text_name}")
                self.logger.log(f"    Signal: {signal}")


    def __save_model_signal(self, model_name, signal):
        """
        Saves given model signal into a file
        """
        file_name = f"{model_name}-signals.json"
        path = self.data_path / self.signals_folder / file_name
        save_to_json(signal, path, f"{model_name} Signal data")
