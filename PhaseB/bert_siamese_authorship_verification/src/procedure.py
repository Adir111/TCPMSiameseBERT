import tensorflow as tf
import gc
from transformers import TFBertModel, BertTokenizer
from transformers import logging as tf_logging
from huggingface_hub import snapshot_download
from huggingface_hub.utils import logging as hf_logging
from pathlib import Path

from .data_loader import DataLoader
from .preprocess import Preprocessor
from .trainer import Trainer
from .model import SiameseBertModel
from .signal_generation import SignalGeneration
from .distance_manager import SignalDistanceManager
from PhaseB.bert_siamese_authorship_verification.utilities import DataVisualizer, increment_last_iteration, \
    artifact_file_exists

# from src.dtw import compute_dtw_distance
# from src.isolation_forest import AnomalyDetector
# from src.clustering import perform_kmedoids_clustering

tf.get_logger().setLevel('ERROR')
hf_logging.set_verbosity_error()
tf_logging.set_verbosity_error()

class Procedure:
    _instance = None

    def __new__(cls, config, logger):
        if cls._instance is None:
            cls._instance = super(Procedure, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config, logger):
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

        self._initialized = True



    # ============================================ Utils ============================================

    def __get_pairs_info(self):
        impostor_pairs_data = self.data_loader.get_pairs()
        impostor_pairs = impostor_pairs_data["pairs"]
        last_iteration_training = impostor_pairs_data["last_iteration_training"]
        last_iteration_signal = impostor_pairs_data["last_iteration_signal"]

        return impostor_pairs, last_iteration_training, last_iteration_signal


    def __load_tokenizer_and_model(self, impostor_name):
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
        impostor_pairs, starting_iteration, _ = self.__get_pairs_info()

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
                f"Training model index {idx} for impostor pair: {impostor_1} and {impostor_2}")
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


    def run_classification_procedure(self):
        # ========= Signal Generation Phase =========
        signal_generator = SignalGeneration(self.config, self.logger)
        impostor_pairs, _, starting_iteration = self.__get_pairs_info()
        self.logger.info(f"Loading {len(impostor_pairs)} pretrained models for classification.")
        signal_generator.load_shakespeare_preprocessed_texts()

        for idx, (impostor_1, impostor_2) in enumerate(impostor_pairs[starting_iteration:], start=starting_iteration):
            self.logger.log("__________________________________________________________________________________________________")
            self.logger.info(f"Generating signal index {idx} for impostor pair: {impostor_1} and {impostor_2}")

            loaded_model = self.__load_trained_network(impostor_1, impostor_2)
            model_name = loaded_model.model_name
            classifier = loaded_model.get_encoder_classifier()

            self.logger.info(f"Generating signals from model: {model_name}...")
            signal_generator.generate_signals_for_preprocessed_texts(classifier, model_name)
            signal_generator.save_model_signal(model_name)

            self.logger.info(f"Model index {idx} signal generation complete.")
            increment_last_iteration(self.config, False)

        signal_generator.print_all_signals()

    def run_distance_matrix_generation(self):
        """
        Runs distance matrix generation for all models that have signals generated.
        """
        self.logger.info("Starting distance matrix generation procedure...")
        signal_processor = SignalDistanceManager(config=self.config, logger=self.logger)

        impostor_pairs, _, starting_iteration = self.__get_pairs_info()

        for idx, (impostor_1, impostor_2) in enumerate(impostor_pairs[starting_iteration:], start=starting_iteration):
            model_name = f"{impostor_1}_{impostor_2}"
            sanitized_model_name = SiameseBertModel.sanitize_artifact_name(model_name)
            self.logger.info(f"Processing distance matrix for model: {sanitized_model_name}")
            signal_processor.compute_distance_matrix_for_model(sanitized_model_name)
            self.logger.info(f"✓ Distance matrix for {sanitized_model_name} completed and saved.")

        self.logger.info("✅ All distance matrices generated.")
