import numpy as np
from pathlib import Path
from dtaidistance import dtw

from .data_loader import DataLoader

from PhaseB.bert_siamese_authorship_verification.utilities import save_to_json


class SignalDistanceManager:
    _instance = None

    def __new__(cls, config, logger):
        if cls._instance is None:
            cls._instance = super(SignalDistanceManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, config, logger):
        if hasattr(self, "_initialized") and self._initialized:
            return  # Prevent re-initialization in singleton
        self._initialized = True

        self.logger = logger
        self.data_loader = DataLoader(config)
        self.output_path = Path(config['data']['organised_data_folder_path']) / config['data']['output_distance_folder']
        self.chunks_per_batch = config['model']['chunk_to_batch_ratio']

        self.output_path.mkdir(parents=True, exist_ok=True)


    def compute_distance_matrix_for_model(self, model_name):
        self.logger.info(f"Computing distance matrix for model: {model_name}")

        # Load signal data
        model_signals = self.data_loader.get_model_signals(model_name)

        # Filter and batch-average signals
        processed_signals, included_text_names = self.__batch_average_signals(model_signals)

        if not processed_signals:
            self.logger.warn("No valid signals to process.")
            return

        # Create distance matrix
        distance_matrix = self.__create_dtw_distance_matrix(processed_signals)

        # Save results
        self.__save_results(processed_signals, distance_matrix, included_text_names, model_name)
        self.logger.info(f"Finished computing distance matrix for {model_name}")

    def __batch_average_signals(self, signals_dict):
        processed = {}
        included_text_names = []

        for text_name, signal in signals_dict.items():
            batched = []

            # Create batches of size self.chunks_per_batch
            signal_length = len(signal)
            for i in range(0, signal_length - self.chunks_per_batch + 1, self.chunks_per_batch):
                chunk = signal[i:i + self.chunks_per_batch]
                avg = sum(chunk) / len(chunk)
                batched.append(avg)

            # Skip signals with fewer than 2 batches
            if len(batched) < 2:
                continue
            included_text_names.append(text_name)

            processed[text_name] = batched

        return processed, included_text_names

    @staticmethod
    def __create_dtw_distance_matrix(signals_dict):
        keys = list(signals_dict.keys())
        m = len(keys)
        dist_mat = np.zeros((m, m))

        for i, key1 in enumerate(keys):
            for j, key2 in enumerate(keys):
                dist_mat[i, j] = dtw.distance(signals_dict[key1], signals_dict[key2])

        return dist_mat

    def __save_results(self, signals_dict, distance_matrix, included_text_names, model_name):
        # Create subfolder for this model
        model_output_dir = self.output_path / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        # Define file paths (without repeating model_name in the filename)
        signal_file_path = model_output_dir / f"signals.json"
        matrix_file_path = model_output_dir / f"distance_matrix.json"
        names_file_path = model_output_dir / "included_text_names.json"

        save_to_json(signals_dict, signal_file_path, f"Batched Signals ({model_name})")
        save_to_json(distance_matrix.tolist(), matrix_file_path, f"DTW ({model_name})")
        save_to_json(included_text_names, names_file_path, f"Included Text Names ({model_name})")
