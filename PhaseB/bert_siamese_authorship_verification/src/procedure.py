import numpy as np
import tensorflow as tf
import gc
from transformers import TFBertModel, BertTokenizer
from huggingface_hub import snapshot_download
from pathlib import Path

from .data_loader import DataLoader
from .preprocess import Preprocessor
from .trainer import Trainer
from .model import SiameseBertModel
from PhaseB.bert_siamese_authorship_verification.utilities import DataVisualizer, increment_last_iteration, artifact_file_exists

# from src.dtw import compute_dtw_distance
# from src.isolation_forest import AnomalyDetector
# from src.clustering import perform_kmedoids_clustering

tf.get_logger().setLevel('ERROR')


class Procedure:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.general_preprocessor = Preprocessor(config=config)  # Uses non-fine-tuned BERT tokenizer, just for utilities...
        self.data_visualizer = DataVisualizer(logger)
        self.chunks_per_batch = config['model']['chunk_to_batch_ratio']
        self.training_batch_size = config['training']['training_batch_size']
        self.data_loader = DataLoader(config=config)
        self.trained_networks = []
        self.model_creator = None

    def __get_pairs_info(self, impostors_names):
        impostor_pairs_data = self.data_loader.get_pairs(impostors_names)
        impostor_pairs = impostor_pairs_data["pairs"]
        last_iteration = impostor_pairs_data["last_iteration"]

        return impostor_pairs, last_iteration

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

        impostor_1_chunks, impostor_2_chunks = self.general_preprocessor.equalize_chunks([impostor_1_chunks, impostor_2_chunks])

        # Log after stabilizing
        self.logger.info(f"After equalization: {impostor_1[0]} - {len(impostor_1_chunks)} chunks")
        self.logger.info(f"After equalization: {impostor_2[0]} - {len(impostor_2_chunks)} chunks")

        self.logger.info("✅ Preprocessing stage has been completed!")
        print("----------------------")
        return impostor_1_chunks, impostor_2_chunks

    def __load_tokenizer_and_model(self, impostor_name):
        model_path = Path(self.config['data']['fine_tuned_bert_model_path'])
        repo_id = self.config['bert']['repository']
        hf_model_id = f"{repo_id}/{impostor_name}"

        # Try Downloading from Hugging Face Hub
        try:
            self.logger.log(f"Downloading model from Hugging Face Hub: {hf_model_id}")
            # Download only the subfolder
            snapshot_path = snapshot_download(
                repo_id,
                allow_patterns=[f"{impostor_name}/*"],
                local_dir=model_path,
                local_dir_use_symlinks=False
            )
            self.logger.log(f"Model downloaded successfully: {hf_model_id}")

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

    def __training_stage(self, model_creator, impostor_1_preprocessed, impostor_2_preprocessed):
        print("----------------------")
        self.logger.info("Starting training stage...")

        trainer = Trainer(self.config, self.logger, model_creator, self.training_batch_size)

        x_train, y_train, x_test, y_test = self.general_preprocessor.create_xy(impostor_1_preprocessed,
                                                                               impostor_2_preprocessed)
        history = trainer.train(x_train, y_train, x_test, y_test)

        self.logger.info("✅ Training stage has been completed!")

        print("----------------------")
        return history

    def __generate_signals_for_text(self, shakespearian_text):
        text_name = shakespearian_text['text_name']
        chunks_list, chunks_tokens_count = self.general_preprocessor.preprocess([shakespearian_text['text']])
        text_chunks = {
            "input_ids": np.stack([c["input_ids"].numpy().squeeze(0) for c in chunks_list]),
            "attention_mask": np.stack([c["attention_mask"].numpy().squeeze(0) for c in chunks_list]),
            "token_type_ids": np.stack([c["token_type_ids"].numpy().squeeze(0) for c in chunks_list]),
        }

        self.logger.info(
            f"Text '{text_name}' has been preprocessed into {len(chunks_list)} chunks with {chunks_tokens_count} tokens.")

        all_signals = []
        for model_creator in self.trained_networks:
            model_name = model_creator.model_name
            classifier = model_creator.get_encoder_classifier()
            self.logger.info(f"Generating signal from model: {model_name}...")

            predictions = np.asarray(classifier.predict({
                "input_ids": text_chunks['input_ids'],
                "attention_mask": text_chunks['attention_mask'],
                "token_type_ids": text_chunks['token_type_ids']
            }))

            binary_outputs = (predictions >= 0.5).astype(int)
            binary_outputs = binary_outputs.flatten().tolist()
            self.logger.log(f"[INFO] Predictions: {predictions}")
            self.logger.log(f"[INFO] Rounded up predictions: {binary_outputs}")

            # Aggregate scores into signal chunks
            signal = [np.mean(binary_outputs[i:i + self.chunks_per_batch]) for i in
                      range(0, len(binary_outputs), self.chunks_per_batch)]
            self.logger.log(f"[INFO] Signal representation: {signal}")

            all_signals.append(signal)
            self.logger.info(
                f"Signal generated for text: {text_name} by model: {model_name}")

            self.data_visualizer.display_signal_plot(signal, text_name, model_name)

    def run_training_procedure(self):
        impostors_names = self.data_loader.get_impostors_name_list()
        impostor_pairs, starting_iteration = self.__get_pairs_info(impostors_names)

        self.logger.info(f"Batch size is {self.training_batch_size}")

        # ========= Training Phase =========
        for idx in range(starting_iteration, len(impostor_pairs)):
            skip_training = False
            impostor_pair = impostor_pairs[idx]
            impostor_1 = impostor_pair[0]
            impostor_2 = impostor_pair[1]
            model_name = f"{impostor_1}_{impostor_2}"

            tokenizer1, bert_model1 = self.__load_tokenizer_and_model(impostor_1)
            tokenizer2, bert_model2 = self.__load_tokenizer_and_model(impostor_2)
            preprocessor1 = Preprocessor(config=self.config, tokenizer=tokenizer1)
            preprocessor2 = Preprocessor(config=self.config, tokenizer=tokenizer2)

            branch_1_weights_exist = artifact_file_exists(
                project_name=self.config['wandb']['project'],
                artifact_name=f"{self.config['wandb']['artifact_name']}-{impostor_1.replace(' ', '_').replace('/', '_')}:latest",
                file_path="branch_weights.h5"
            )
            branch_2_weights_exist = artifact_file_exists(
                project_name=self.config['wandb']['project'],
                artifact_name=f"{self.config['wandb']['artifact_name']}-{impostor_2.replace(' ', '_').replace('/', '_')}:latest",
                file_path="branch_weights.h5"
            )

            if branch_1_weights_exist and branch_2_weights_exist:
                skip_training = True

            model_creator = SiameseBertModel(config=self.config, logger=self.logger, impostor_1_name=impostor_1, impostor_2_name=impostor_2, use_pretrained_weights=skip_training)
            model_creator.build_siamese_model(bert_model1, bert_model2)

            if skip_training:
                self.logger.info(
                    f"[✓] Loaded existing weights for '{model_name}'. "
                    f"Skipping training."
                )
                self.trained_networks.append(model_creator)
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

            self.trained_networks.append(model_creator)

            self.logger.info(f"Model index {idx} training complete.")
            self.data_visualizer.display_accuracy_plot(history, model_name)
            self.data_visualizer.display_loss_plot(history, model_name)

            increment_last_iteration(self.config)

        self.logger.info(f"Finished training {len(self.trained_networks)} models successfully!")


    def run_classification_procedure(self):
        # ========= Signal Generation Phase =========
        tested_collection_texts = self.data_loader.get_shakespeare_data()
        for text in tested_collection_texts:
            self.logger.info(f"Processing text: {text['text_name']}")
            self.__generate_signals_for_text(text)
