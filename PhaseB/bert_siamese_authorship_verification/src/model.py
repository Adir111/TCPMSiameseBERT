"""
Defines the SiameseBertModel class for creating a Siamese neural network architecture
using two BERT-based branches combined with CNN and BiLSTM layers, trained to produce
embedding vectors for similarity comparison.

The model supports:
- Building the Siamese model architecture
- Generating a classifier from encoder branches
- Loading and saving model weights using Weights & Biases artifacts
- Utility methods for distance calculation and artifact name sanitization
"""

import os
import io
import re

import unicodedata
import wandb
import tensorflow as tf
from contextlib import redirect_stdout
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Bidirectional, Dropout, LSTM, Lambda


class SiameseBertModel:
    """
    Implements a Siamese neural network model using two BERT-based branches combined with CNN and BiLSTM layers.

    The model encodes two input sequences into embeddings and computes their similarity via
    a Euclidean distance layer followed by a sigmoid output for binary similarity classification.
    """

    def __init__(self, config, logger, impostor_1_name, impostor_2_name, use_pretrained_weights=False):
        """
        Initializes the SiameseBertModel instance.

        Args:
            config (dict): Configuration dictionary with model parameters.
            logger (logging.Logger): Logger for info and debug messages.
            impostor_1_name (str): Name identifier for the first branch.
            impostor_2_name (str): Name identifier for the second branch.
            use_pretrained_weights (bool, optional): Flag to load pretrained weights. Defaults to False.
        """
        self.config = config
        self.logger = logger
        self.use_pretrained_weights = use_pretrained_weights
        self.model_name = f"{impostor_1_name}_{impostor_2_name}"
        self._impostor_1_name = impostor_1_name
        self._impostor_2_name = impostor_2_name

        self.bilstm_units = config['model']['bilstm']['units']
        self.bilstm_dropout = config['model']['bilstm']['dropout']

        self.filters = config['model']['cnn']['filters']
        self.pool_size = config['model']['cnn']['pool_size']
        self.padding = config['model']['cnn']['padding']
        self.kernel_size = config['model']['cnn']['kernel_size']

        self.in_features = config['model']['fc']['in_features']
        self.out_features = config['model']['fc']['out_features']

        self.chunk_size = config['model']['chunk_size']

        self.model = None  # Will be set in build_model()
        self._branch_1 = None
        self._branch_2 = None
        self._similarity_head = None


    @staticmethod
    def get_model_summary_string(model):
        """
        Returns the string summary of a given Keras model.

        Args:
            model (tf.keras.Model): The model to summarize.

        Returns:
            str: The string representation of the model summary.
        """
        with io.StringIO() as buf, redirect_stdout(buf):
            model.summary()
            return buf.getvalue()


    @staticmethod
    def euclidean_distance(vec_1, vec_2):
        """
        Computes the Euclidean distance between two vectors.

        Args:
            vec_1 (tf.Tensor): First input vector tensor.
            vec_2 (tf.Tensor): Second input vector tensor.

        Returns:
            tf.Tensor: Tensor containing the Euclidean distance for each batch element.
        """
        return Lambda(lambda tensors: tf.sqrt(
            tf.reduce_sum(tf.square(tensors[0] - tensors[1]), axis=1, keepdims=True) + 1e-6
        ))([vec_1, vec_2])


    @staticmethod
    def sanitize_artifact_name(name):
        """
        Sanitizes artifact names by normalizing unicode and removing invalid characters.

        Args:
            name (str): The artifact name to sanitize.

        Returns:
            str: Sanitized artifact name safe for W&B artifact usage.
        """
        # Normalize Unicode characters (e.g., "ë" → "e")
        name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
        # Remove characters not in [a-zA-Z0-9._-]
        name = re.sub(r'[^a-zA-Z0-9._-]', '', name)

        name.replace(" ", "_").replace("/", "_")
        return name


    @staticmethod
    def __get_weight_path(artifact_name):
        """
        Downloads the model weights artifact and returns the local path to the weights file.

        Args:
            artifact_name (str): The Weights & Biases artifact name (including version).

        Returns:
            str: Local file path to the downloaded model weights.
        """
        artifact = wandb.use_artifact(artifact_name, type="model")
        artifact_dir = artifact.download()
        return os.path.join(artifact_dir, "model_weights.h5")


    @staticmethod
    def __upload_weights_to_artifact(branch_weights_path, artifact_name):
        """
        Uploads model weights to a Weights & Biases artifact.

        Args:
            branch_weights_path (str): Path to the local weights file.
            artifact_name (str): Name of the artifact to create/upload.
        """
        def sanitize_artifact_name(name):
            return name.replace(" ", "_").replace("/", "_")

        artifact = wandb.Artifact(name=artifact_name, type="model")
        artifact.add_file(branch_weights_path)
        wandb.log_artifact(artifact)


    def _build_siamese_branch(self, bert_model):
        """
        Builds a single Siamese branch using BERT, CNN, BiLSTM, and fully-connected layers.

        The architecture is:
            Inputs → BERT output → several Conv1D + MaxPooling1D → 2×BiLSTM → Dense → Dropout → Dense embedding

        Returns:
            tf.keras.Model: A Keras model that outputs an embedding vector of dimension `self.out_features`.
        """
        # ------------------------------------------------------------------ #
        # 1.  Inputs (int32) ------------------------------------------------ #
        # ------------------------------------------------------------------ #
        input_ids = Input(shape=(self.chunk_size,), dtype=tf.int32, name="input_ids")
        attention_mask = Input(shape=(self.chunk_size,), dtype=tf.int32, name="attention_mask")
        token_type_ids = Input(shape=(self.chunk_size,), dtype=tf.int32, name="token_type_ids")

        # ------------------------------------------------------------------ #
        # 2.  Contextual token embeddings from fine-tuned BERT -------------- #
        # ------------------------------------------------------------------ #
        # outputs[0] = (batch, seq_len, 768)
        bert_output = bert_model(
            {"input_ids": input_ids,
             "attention_mask": attention_mask,
             "token_type_ids": token_type_ids})[0]

        # ------------------------------------------------------------------ #
        # 3.  CNN + BiLSTM stack (≈ rcnna) ---------------------------------- #
        # ------------------------------------------------------------------ #
        cnn_lstm = Sequential(name="cnn_bilstm_stack")

        # --- 3.a several Conv1D + MaxPool blocks -------------------------- #
        for k in self.kernel_size:
            cnn_lstm.add(
                Conv1D(filters=self.filters,
                       kernel_size=k,
                       padding=self.padding,
                       activation='relu')
            )
            cnn_lstm.add(MaxPooling1D(pool_size=self.pool_size))

        # --- 3.b two BiLSTM layers (first returns seq, second not) -------- #
        cnn_lstm.add(
            Bidirectional(
                LSTM(units=self.bilstm_units,
                     return_sequences=True),
                merge_mode='concat')
        )
        cnn_lstm.add(
            Bidirectional(
                LSTM(units=self.bilstm_units,
                     return_sequences=False,  # 2-D output
                     go_backwards=True),
                merge_mode='concat')
        )

        # --- 3.c Fully-connected head ------------------------------------ #
        cnn_lstm.add(Dense(self.in_features, activation='relu'))
        cnn_lstm.add(Dropout(self.bilstm_dropout))

        # ------------------------------------------------------------------ #
        # 4.  Produce the *embedding* that the Siamese distance will use ---- #
        # ------------------------------------------------------------------ #
        embedding = Dense(self.out_features,
                          activation=None,  # linear
                          name="embedding")

        x = cnn_lstm(bert_output)  # (batch, in_features)
        outputs = embedding(x)  # (batch, out_features)

        return Model(inputs=[input_ids,
                             attention_mask,
                             token_type_ids],
                     outputs=outputs)


    def build_siamese_model(self, bert_model_1, bert_model_2, print_summary=True):
        """
        Builds the complete Siamese model with two BERT branches.

        This model is intended for training and does not load pretrained weights.

        Args:
            bert_model_1 (tf.keras.Model): First BERT encoder model.
            bert_model_2 (tf.keras.Model): Second BERT encoder model.
            print_summary (bool, optional): Whether to log the model summary. Defaults to True.

        Returns:
            tf.keras.Model: The compiled Siamese model.
        """
        self.logger.info(f"Started building model {self.model_name}...")

        input_ids_1 = Input(shape=(self.chunk_size,), dtype=tf.int32, name="input_ids_1")
        attention_mask_1 = Input(shape=(self.chunk_size,), dtype=tf.int32, name="attention_mask_1")
        token_type_ids_1 = Input(shape=(self.chunk_size,), dtype=tf.int32, name="token_type_ids_1")

        input_ids_2 = Input(shape=(self.chunk_size,), dtype=tf.int32, name="input_ids_2")
        attention_mask_2 = Input(shape=(self.chunk_size,), dtype=tf.int32, name="attention_mask_2")
        token_type_ids_2 = Input(shape=(self.chunk_size,), dtype=tf.int32, name="token_type_ids_2")

        self._branch_1 = self._build_siamese_branch(bert_model_1)
        self._branch_2 = self._build_siamese_branch(bert_model_2)
        self._similarity_head = Dense(1, "sigmoid", name="similarity")

        out1 = self._branch_1([input_ids_1, attention_mask_1, token_type_ids_1])
        out2 = self._branch_2([input_ids_2, attention_mask_2, token_type_ids_2])

        distance = self.euclidean_distance(out1, out2)
        output = self._similarity_head(distance)

        self.model = Model(
            inputs=[
                input_ids_1, attention_mask_1, token_type_ids_1,
                input_ids_2, attention_mask_2, token_type_ids_2
            ],
            outputs=output
        )
        if print_summary:
            model_summary = self.get_model_summary_string(self.model)
            self.logger.log(model_summary)
        self.logger.info(f"Finished building model {self.model_name}...")
        return self.model


    def get_encoder_classifier(self):
        """
        Creates a classifier model from the Siamese encoder branches.

        This model accepts a single input and produces a similarity score
        by encoding the input twice (through both branches) and computing
        their distance.

        Returns:
            tf.keras.Model: Classifier model using the encoder branches.

        Raises:
            RuntimeError: If Siamese branches are not initialized.
        """
        if self._branch_1 is None or self._branch_2 is None:
            raise RuntimeError("You must call build_siamese_model() first to initialize the branches.")

        if self.use_pretrained_weights:
            self.logger.log("Loading pre-trained branch weights from artifacts...")
            self.__load_weights()

        input_ids = Input(shape=(self.chunk_size,), dtype=tf.int32, name="input_ids")
        attention_mask = Input(shape=(self.chunk_size,), dtype=tf.int32, name="attention_mask")
        token_type_ids = Input(shape=(self.chunk_size,), dtype=tf.int32, name="token_type_ids")

        out1 = self._branch_1([input_ids, attention_mask, token_type_ids])
        out2 = self._branch_2([input_ids, attention_mask, token_type_ids])

        distance = self.euclidean_distance(out1, out2)
        output = self._similarity_head(distance)

        return Model(
            inputs=[input_ids, attention_mask, token_type_ids],
            outputs=output
        )


    def __load_weights(self):
        """
        Loads pretrained weights from Weights & Biases artifacts into the current model.
        """
        base_artifact_name = self.config["wandb"]["artifact_name"]
        artifact_path = f"{base_artifact_name}-{self.sanitize_artifact_name(self.model_name)}:latest"

        model_path = self.__get_weight_path(artifact_path)

        self.model.load_weights(model_path)

        self.logger.log(f"Loaded weights from artifact: {artifact_path}")


    def save_weights(self):
        """
        Saves the model weights locally and uploads them to Weights & Biases artifacts.
        """
        base_output_dir = self.config['data']['trained_siamese_path']
        base_artifact_name = self.config['wandb']['artifact_name']

        output_dir = f"{base_output_dir}/{self.model_name}"
        os.makedirs(output_dir, exist_ok=True)

        model_path = os.path.join(output_dir, "model_weights.h5")

        self.model.save_weights(model_path)

        artifact_path = f"{base_artifact_name}-{self.sanitize_artifact_name(self.model_name)}"

        self.__upload_weights_to_artifact(model_path, artifact_path)

        self.logger.log(f"Saved and logged artifacts: {artifact_path}")
