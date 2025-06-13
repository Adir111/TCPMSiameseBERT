"""
Procedure module for managing the entire workflow of the Siamese BERT authorship verification system.

This module defines the Procedure class, which orchestrates the stages of training, signal generation,
distance matrix computation, anomaly detection via isolation forest, and clustering. It manages loading
models and tokenizers, preprocessing data, training siamese networks, generating signals from models,
computing distance matrices using DTW, detecting anomalies, and performing clustering on the results.

The Procedure class implements a singleton pattern to ensure a single shared instance during runtime.

Key functionalities:
- Loading and filtering impostor pairs
- Loading pretrained BERT models and tokenizers
- Preprocessing input texts for siamese training
- Training siamese BERT models for authorship verification
- Generating signal data for analysis
- Computing distance matrices using Dynamic Time Warping (DTW)
- Detecting anomalies with Isolation Forest models
- Performing clustering on anomaly detection results

Usage:
Instantiate Procedure with a config dict and logger instance, then run the desired procedures in sequence.

Example:
    procedure = Procedure(config, logger)
    procedure.run_training_procedure()
    procedure.run_signal_generation_procedure()
    procedure.run_distance_matrix_generation()
    procedure.run_isolation_forest_procedure()
    procedure.run_clustering_procedure()
"""

import numpy as np
import tensorflow as tf
import gc
from transformers import TFBertModel, BertTokenizer
from transformers import logging as tf_logging
from huggingface_hub import snapshot_download
from huggingface_hub.utils import logging as hf_logging
from pathlib import Path
import warnings

from .data_loader import DataLoader
from .preprocess import Preprocessor
from .trainer import Trainer
from .model import SiameseBertModel
from .signal_generation import SignalGeneration
from .distance_manager import SignalDistanceManager
from .isolation_forest import DTWIsolationForest
from .clustering import Clustering
from PhaseB.bert_siamese_authorship_verification.utilities import DataVisualizer, increment_last_iteration, \
    artifact_file_exists

# from src.dtw import compute_dtw_distance
# from src.isolation_forest import AnomalyDetector
# from src.clustering import perform_kmedoids_clustering

tf.get_logger().setLevel('ERROR')
hf_logging.set_verbosity_error()
tf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning)


class Procedure:
    """
    Singleton class to manage the authorship verification procedure workflow.

    This class handles:
    - Training siamese BERT models for pairs of impostors.
    - Generating signal representations from trained models.
    - Computing distance matrices based on generated signals.
    - Running anomaly detection using Isolation Forest.
    - Performing clustering on anomaly detection results.
    """

    _instance = None

    def __new__(cls, config, logger):
        """
        Implements the singleton pattern to ensure only one instance of the Procedure class.

        Args:
            config (dict): Configuration parameters (unused in __new__).
            logger (Logger): Logger instance (unused in __new__).

        Returns:
            Procedure: Singleton instance of the Procedure class.
        """
        if cls._instance is None:
            cls._instance = super(Procedure, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config, logger):
        """
        Initializes the Procedure instance if not already initialized.

        Args:
            config (dict): Configuration parameters.
            logger (Logger): Logger for informational output.
        """
        if self._initialized:
            return  # Avoid reinitialization

        self.config = config
        self.logger = logger
        self.general_preprocessor = Preprocessor(
            config=config)  # Uses non-fine-tuned BERT tokenizer, just for utilities...
        self.data_visualizer = DataVisualizer(config['wandb']['enabled'], logger)
        self.training_batch_size = config['training']['training_batch_size']
        self.load_pretrained_model = config['training']['load_pretrained_model']
        self.data_loader = DataLoader(config=config)
        self.trained_networks = {}
        self.model_creator = None
        self.clustering_increment = config['clustering']['increment']
        self.should_skip_generated_signals = config['procedure']['should_skip_generated_signals']
        self.should_skip_generated_dtw = config['procedure']['should_skip_generated_dtw']

        self._initialized = True



    # ============================================ Utils ============================================

    def __get_pairs_info(self, should_filter=True):
        """
        Retrieves impostor pairs information from the data loader, optionally filtering out skipped models.

        Args:
            should_filter (bool): Whether to filter out models specified to skip.

        Returns:
            tuple:
                filtered_pairs (list): List of filtered impostor pairs if filtering, else all pairs.
                last_iteration_training (int): The last completed iteration index for training.
                last_iteration_signal (int): The last completed iteration index for signal generation.
        """
        self.logger.info("==================== Pairs Info ====================")
        impostor_pairs_data = self.data_loader.get_pairs()
        impostor_pairs = impostor_pairs_data["pairs"]
        models_to_skip_raw = impostor_pairs_data.get("models_to_skip", [])
        last_iteration_training = impostor_pairs_data["last_iteration_training"]
        last_iteration_signal = impostor_pairs_data["last_iteration_signal"]

        # Convert list to set for faster lookup
        models_to_skip = set(models_to_skip_raw)
        filtered_pairs = []
        if should_filter:
            for pair in impostor_pairs:
                model_name = f"{pair[0]}_{pair[1]}"
                if model_name in models_to_skip:
                    self.logger.info(f"Skipping model: {model_name}")
                    continue
                filtered_pairs.append(pair)

            self.logger.info(f"Total pairs before filtering: {len(impostor_pairs)}")
            self.logger.info(f"Total pairs after filtering: {len(filtered_pairs)}")
        else:
            self.logger.info("==================== Finished Pairs Info ====================")
            return impostor_pairs, last_iteration_training, last_iteration_signal

        self.logger.info("==================== Finished Pairs Info ====================")
        return filtered_pairs, last_iteration_training, last_iteration_signal


    def __load_tokenizer_and_model(self, impostor_name):
        """
        Loads a pretrained BERT tokenizer and model for the specified impostor.

        Tries to download the model from Hugging Face Hub under the configured repository.

        Args:
            impostor_name (str): The name of the impostor model to load.

        Returns:
            tuple: (tokenizer, model) where tokenizer is BertTokenizer and model is TFBertModel.

        Raises:
            RuntimeError: If model loading fails.
        """
        model_path = Path(self.config['data']['fine_tuned_bert_model_path'])
        repo_id = self.config['bert']['repository']
        hf_model_id = f"{repo_id}/{impostor_name}"

        # Try Downloading from Hugging Face Hub
        try:
            self.logger.info(f"Downloading model from Hugging Face Hub: {hf_model_id}")
            # Download only the subfolder
            snapshot_path = snapshot_download(
                repo_id,
                allow_patterns=[f"{impostor_name}/*"],
                local_dir=model_path,
                local_dir_use_symlinks=False
            )
            self.logger.info(f"Model downloaded successfully: {hf_model_id}")

            download_path = Path(snapshot_path) / impostor_name

            tokenizer = BertTokenizer.from_pretrained(download_path)
            model = TFBertModel.from_pretrained(download_path)
            model.trainable = False  # Freeze the BERT's weights!!! Super important.

            return tokenizer, model
        except Exception as e:
            self.logger.log(f"Failed to load model: {hf_model_id}. Error: {e}")
            raise RuntimeError(
                f"Failed to load BERT model: {hf_model_id}. Error: {e}"
            )


    def __load_trained_network(self, impostor_1, impostor_2):
        """
        Loads a trained Siamese BERT model for a given impostor pair if available.

        Checks if the model weights exist and loads the pretrained network,
        caching it in the trained_networks dictionary.

        Args:
            impostor_1 (str): First impostor name.
            impostor_2 (str): Second impostor name.

        Returns:
            SiameseBertModel or None: Loaded model or None if weights missing.
        """
        # ========= Signal Generation Phase =========
        model_name = f"{impostor_1}_{impostor_2}"
        sanitized_model_name = SiameseBertModel.sanitize_artifact_name(model_name)

        # Skip if model already loaded
        if model_name in self.trained_networks:
            self.logger.info(f"Model for {model_name} already loaded. Skipping.")
            return self.trained_networks[model_name]

        self.logger.info(f"Loading model for impostor pair: {model_name}")

        # Load tokenizers and models
        tokenizer1, bert_model1 = self.__load_tokenizer_and_model(impostor_1)
        tokenizer2, bert_model2 = self.__load_tokenizer_and_model(impostor_2)

        # Check that both weights exist
        artifact_name = f"{self.config['wandb']['artifact_name']}-{sanitized_model_name}:latest"

        weights_exist = artifact_file_exists(
            project_name=self.config['wandb']['project'],
            artifact_name=artifact_name,
            file_path="model_weights.h5"
        )

        if not weights_exist:
            self.logger.warn(f"Skipping model {model_name} due to missing weights.")
            return None

        # Build Siamese model with pretrained weights
        model_creator = SiameseBertModel(
            config=self.config,
            logger=self.logger,
            impostor_1_name=impostor_1,
            impostor_2_name=impostor_2,
            use_pretrained_weights=True
        )
        model_creator.build_siamese_model(bert_model1, bert_model2, False)

        # Add to trained networks
        self.logger.info(f"✓ Loaded model for {model_name}.")
        return model_creator


    # ============================================ Training Stages ============================================

    def __preprocessing_stage(self, impostor_1: tuple, impostor_2: tuple):
        """
        Executes the preprocessing stage for two impostor datasets.

        Loads raw texts, preprocesses into chunks and tokens, then equalizes chunk counts
        between impostors for balanced training.

        Args:
            impostor_1 (tuple): Tuple containing impostor name and Preprocessor instance.
            impostor_2 (tuple): Tuple containing impostor name and Preprocessor instance.

        Returns:
            tuple: Two lists of preprocessed chunks corresponding to impostor_1 and impostor_2.
        """
        print("----------------------")
        self.logger.info("Starting preprocessing stage...")

        def __load_and_preprocess(impostor: tuple):
            (impostor_name, preprocessor) = impostor
            impostor_texts = self.data_loader.get_impostor_texts_by_name(impostor_name)
            impostor_chunks, impostor_tokens_count = preprocessor.preprocess(impostor_texts)
            self.logger.info(
                f"Before equalization: {impostor_name} - {len(impostor_chunks)} chunks with {impostor_tokens_count} tokens")
            return impostor_chunks, impostor_tokens_count

        impostor_1_chunks, impostor_1_tokens_count = __load_and_preprocess(impostor_1)
        impostor_2_chunks, impostor_2_tokens_count = __load_and_preprocess(impostor_2)

        impostor_1_chunks, impostor_2_chunks = self.general_preprocessor.equalize_chunks(
            [impostor_1_chunks, impostor_2_chunks])

        # Log after stabilizing
        self.logger.info(f"After equalization: {impostor_1[0]} - {len(impostor_1_chunks)} chunks")
        self.logger.info(f"After equalization: {impostor_2[0]} - {len(impostor_2_chunks)} chunks")

        self.logger.info("✅ Preprocessing stage has been completed!")
        print("----------------------")
        return impostor_1_chunks, impostor_2_chunks


    def __training_stage(self, model_creator, impostor_1_preprocessed, impostor_2_preprocessed):
        """
        Executes the training stage for a siamese model on preprocessed impostor data.

        Args:
            model_creator (SiameseBertModel): The model instance to train.
            impostor_1_preprocessed (list): Preprocessed chunks of impostor 1.
            impostor_2_preprocessed (list): Preprocessed chunks of impostor 2.

        Returns:
            History: Training history object returned by TensorFlow Keras.
        """
        print("----------------------")
        self.logger.info("Starting training stage...")

        trainer = Trainer(self.config, self.logger, model_creator, self.training_batch_size)

        x_train, y_train, x_test, y_test = self.general_preprocessor.create_xy(
            impostor_1_preprocessed,
            impostor_2_preprocessed
        )
        history = trainer.train(x_train, y_train, x_test, y_test)

        self.logger.info("✅ Training stage has been completed!")
        print("----------------------")
        return history



    # ============================================ Procedures ============================================

    def run_training_procedure(self):
        """
        Runs the full training procedure for all impostor pairs, loading pretrained weights
        when available and skipping training accordingly.

        Saves trained models in the trained_networks cache.

        Logs training progress, visualizes accuracy and loss, and increments iteration tracking.
        """
        impostor_pairs, starting_iteration, _ = self.__get_pairs_info(False)
        total_pairs = len(impostor_pairs)

        self.logger.info(f"Batch size is {self.training_batch_size}")

        # ========= Training Phase =========
        for idx, (impostor_1, impostor_2) in enumerate(impostor_pairs[starting_iteration:], start=starting_iteration):
            skip_training = False
            model_name = f"{impostor_1}_{impostor_2}"
            sanitized_model_name = SiameseBertModel.sanitize_artifact_name(model_name)
            artifact_name = f"{self.config['wandb']['artifact_name']}-{sanitized_model_name}:latest"

            tokenizer1, bert_model1 = self.__load_tokenizer_and_model(impostor_1)
            tokenizer2, bert_model2 = self.__load_tokenizer_and_model(impostor_2)
            preprocessor1 = Preprocessor(config=self.config, tokenizer=tokenizer1)
            preprocessor2 = Preprocessor(config=self.config, tokenizer=tokenizer2)

            if self.load_pretrained_model:
                weights_exist = artifact_file_exists(
                    project_name=self.config['wandb']['project'],
                    artifact_name=artifact_name,
                    file_path="model_weights.h5"
                )

                if weights_exist:
                    skip_training = True

            model_creator = SiameseBertModel(
                config=self.config,
                logger=self.logger,
                impostor_1_name=impostor_1,
                impostor_2_name=impostor_2,
                use_pretrained_weights=skip_training
            )
            model_creator.build_siamese_model(bert_model1, bert_model2)

            if skip_training:
                self.logger.info(
                    f"[✓] Loaded existing weights for '{model_name}'. "
                    f"Skipping training."
                )
                self.trained_networks[model_name] = model_creator
                continue

            self.logger.info(
                f"Training model index {idx + 1}/{total_pairs} for impostor pair: {impostor_1} and {impostor_2}")
            impostor_1_preprocessed, impostor_2_preprocessed = self.__preprocessing_stage((impostor_1, preprocessor1),
                                                                                          (impostor_2, preprocessor2))
            del preprocessor1, preprocessor2
            gc.collect()

            history = self.__training_stage(model_creator, impostor_1_preprocessed, impostor_2_preprocessed)

            del impostor_1_preprocessed, impostor_2_preprocessed
            gc.collect()

            key = f"{impostor_1}_{impostor_2}"
            self.trained_networks[key] = model_creator

            self.logger.info(f"Model index {idx} training complete.")
            self.data_visualizer.display_accuracy_plot(history, model_name)
            self.data_visualizer.display_loss_plot(history, model_name)

            increment_last_iteration(self.config)

        self.logger.info(f"Finished training {len(self.trained_networks)} models successfully!")


    def run_signal_generation_procedure(self):
        """
        Runs signal generation for all impostor pairs using pretrained models.

        Loads Shakespeare preprocessed texts, generates signals from the siamese model's encoder classifier,
        and skips pairs where signals already exist if configured.

        Prompts the user whether to print all generated signals at the end.
        """
        signal_generator = SignalGeneration(self.config, self.logger)
        impostor_pairs, _, starting_iteration = self.__get_pairs_info()
        self.logger.info(f"Loading {len(impostor_pairs)} pretrained models for classification.")
        signal_generator.load_shakespeare_preprocessed_texts()
        total_pairs = len(impostor_pairs)

        for idx, (impostor_1, impostor_2) in enumerate(impostor_pairs[starting_iteration:], start=starting_iteration):
            self.logger.log("__________________________________________________________________________________________________")
            model_name = f"{impostor_1}_{impostor_2}"
            sanitized_model_name = SiameseBertModel.sanitize_artifact_name(model_name)
            if self.should_skip_generated_signals and signal_generator.signal_already_exists(sanitized_model_name):
                self.logger.info(f"Signal for model '{sanitized_model_name}' already exists. Skipping signal generation for {idx + 1}/{total_pairs}.")
                continue

            self.logger.info(f"Generating signal index {idx + 1}/{total_pairs} for impostor pair: {impostor_1} and {impostor_2}")

            loaded_model = self.__load_trained_network(impostor_1, impostor_2)
            if loaded_model is None:
                continue
            classifier = loaded_model.get_encoder_classifier()

            self.logger.info(f"Generating signals from model: {sanitized_model_name}...")
            signal_generator.generate_signals_for_preprocessed_texts(classifier, sanitized_model_name)

            self.logger.info(f"Model index {idx + 1}/{total_pairs} signal generation complete.")
            increment_last_iteration(self.config, False)

        user_input = input("Do you want to print all model signals? (y/n): ").strip().lower()
        if user_input in {"y", "ye", "yes"}:
            signal_generator.print_all_signals()


    def run_distance_matrix_generation(self):
        """
        Runs distance matrix generation for all models with generated signals.
        For each impostor pair model, computes and saves the Dynamic Time Warping (DTW) distance matrix.
        Skips computation for models with existing DTW results if skipping is enabled.
        Logs progress and completion status.
        """
        self.logger.info("Starting distance matrix generation procedure...")
        signal_processor = SignalDistanceManager(config=self.config, logger=self.logger)

        impostor_pairs, _, _ = self.__get_pairs_info()
        total_pairs = len(impostor_pairs)

        for index, (impostor_1, impostor_2) in enumerate(impostor_pairs):
            model_name = f"{impostor_1}_{impostor_2}"
            sanitized_model_name = SiameseBertModel.sanitize_artifact_name(model_name)
            if self.should_skip_generated_dtw and signal_processor.dtw_results_already_exist(sanitized_model_name):
                self.logger.info(f"DTW results for model '{sanitized_model_name}' already exist. Skipping computation for {index + 1}/{total_pairs}.")
                continue

            self.logger.info(f"Processing distance matrix for model: {sanitized_model_name} - {index + 1}/{total_pairs}")
            signal_processor.compute_distance_matrix_for_model(sanitized_model_name)
            self.logger.info(f"✓ Distance matrix for {sanitized_model_name} completed and saved.")

        self.logger.info("✅ All distance matrices generated.")


    def run_isolation_forest_procedure(self):
        """
        Runs anomaly detection using Isolation Forest on DTW distance matrices for all models.
        For each model, analyzes anomalies, logs anomaly ranks, score ranges, and detected anomalies count.
        Saves all models' anomaly scores after processing.
        """
        self.logger.info("🚨 Starting Isolation Forest anomaly detection...")
        anomaly_detector = DTWIsolationForest(config=self.config, logger=self.logger)

        impostor_pairs, _, _ = self.__get_pairs_info()
        total_pairs = len(impostor_pairs)

        for index, (impostor_1, impostor_2) in enumerate(impostor_pairs):
            model_name = f"{impostor_1}_{impostor_2}"
            sanitized_model_name = SiameseBertModel.sanitize_artifact_name(model_name)
            self.logger.info(f"Analyzing anomalies for model: {sanitized_model_name}")

            summa, scores, y_pred_train, rank = anomaly_detector.analyze(sanitized_model_name)

            self.logger.info(f"✅ Model: {sanitized_model_name} - {index + 1}/{total_pairs}")
            self.logger.info(f"   → Anomaly rank (hits in ground truth): {rank}")
            self.logger.info(f"   → Isolation Forest anomaly score range: [{scores.min():.4f}, {scores.max():.4f}]")
            self.logger.info(f"   → Total anomalies detected: {np.array(y_pred_train == -1).sum()}")

        anomaly_detector.save_all_models_scores()


    def run_clustering_procedure(self):
        """
        Runs clustering analysis on the isolation forest anomaly scores.
        Performs clustering optionally with increments.
        Saves clustering results to JSON, generates plots, and logs summaries.
        Iterates over clustering steps to visualize and update state.
        Prints a full summary after all clustering is completed.
        """
        self.logger.info("🔍 Starting clustering procedure...")

        clustering = Clustering(config=self.config, logger=self.logger)
        results = clustering.cluster_results(self.clustering_increment)

        for step_idx, result in enumerate(results):
            suffix = result["suffix"].lstrip("_") or "all_models"
            self.logger.info(f"📈 Visualizing result for: {suffix}")
            clustering.update_state_from_result(result)

            clustering.plot_clustering_results(suffix=suffix)
            clustering.plot_core_vs_outside(suffix=suffix)

        self.logger.info("Printing the clustering summary for all scores")
        clustering.print_full_clustering_summary() # should only print once all is done.
