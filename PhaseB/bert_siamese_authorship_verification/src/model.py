import io
import tensorflow as tf
from transformers import TFBertModel
from contextlib import redirect_stdout
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Bidirectional, Dropout, LSTM, Lambda


class SiameseBertModel:
    def __init__(self, config, logger, model_name):
        self.config = config
        self.logger = logger
        self.model_name = model_name

        self.bilstm_units = self.config['model']['bilstm']['units']
        self.bilstm_dropout = self.config['model']['bilstm']['dropout']

        self.filters = self.config['model']['cnn']['filters']
        self.pool_size = self.config['model']['cnn']['pool_size']
        self.padding = self.config['model']['cnn']['padding']
        self.kernel_size = self.config['model']['cnn']['kernel_size']

        self.in_features = self.config['model']['fc']['in_features']
        self.out_features = self.config['model']['fc']['out_features']

        self.max_len = self.config['bert']['max_sequence_length']

        self.model = None  # Will be set in build_model()
        self._branch_1 = None
        self._branch_2 = None

    @staticmethod
    def get_model_summary_string(model):
        with io.StringIO() as buf, redirect_stdout(buf):
            model.summary()
            return buf.getvalue()

    @staticmethod
    def euclidean_distance(vec_1, vec_2):
        return Lambda(lambda tensors: tf.sqrt(
            tf.reduce_sum(tf.square(tensors[0] - tensors[1]), axis=1, keepdims=True) + 1e-6
        ))([vec_1, vec_2])

    def _build_siamese_branch(self, bert_model):
        """
        Siamese encoder ≈ rcnna():
        [BERT] → [several Conv1D-MaxPool] → 2×BiLSTM → Dense → Dropout → Dense
        ----------------------------------------------------------------------
        Returns: embedding vector of length `self.out_features` (≠ 1!)
        """
        # ------------------------------------------------------------------ #
        # 1.  Inputs (int32) ------------------------------------------------ #
        # ------------------------------------------------------------------ #
        input_ids = Input(shape=(self.max_len,), dtype=tf.int32, name="input_ids")
        attention_mask = Input(shape=(self.max_len,), dtype=tf.int32, name="attention_mask")
        token_type_ids = Input(shape=(self.max_len,), dtype=tf.int32, name="token_type_ids")

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

    def build_siamese_model(self, bert_model_1, bert_model_2):
        self.logger.log(f"Started building model {self.model_name}...")

        input_ids_1 = Input(shape=(self.max_len,), dtype=tf.int32, name="input_ids_1")
        attention_mask_1 = Input(shape=(self.max_len,), dtype=tf.int32, name="attention_mask_1")
        token_type_ids_1 = Input(shape=(self.max_len,), dtype=tf.int32, name="token_type_ids_1")

        input_ids_2 = Input(shape=(self.max_len,), dtype=tf.int32, name="input_ids_2")
        attention_mask_2 = Input(shape=(self.max_len,), dtype=tf.int32, name="attention_mask_2")
        token_type_ids_2 = Input(shape=(self.max_len,), dtype=tf.int32, name="token_type_ids_2")

        self._branch_1 = self._build_siamese_branch(bert_model_1)
        self._branch_2 = self._build_siamese_branch(bert_model_2)

        out1 = self._branch_1([input_ids_1, attention_mask_1, token_type_ids_1])
        out2 = self._branch_2([input_ids_2, attention_mask_2, token_type_ids_2])

        distance = self.euclidean_distance(out1, out2)
        output = Dense(1, activation="sigmoid", name="similarity")(distance)

        self.model = Model(
            inputs=[
                input_ids_1, attention_mask_1, token_type_ids_1,
                input_ids_2, attention_mask_2, token_type_ids_2
            ],
            outputs=output
        )
        model_summary = self.get_model_summary_string(self.model)
        self.logger.log(model_summary)
        self.logger.log(f"Finished building model {self.model_name}...")
        return self.model

    def get_encoder_classifier(self):
        """ Creates a classifier model using the stored encoder branches """
        if self._branch_1 is None or self._branch_2 is None:
            raise RuntimeError("You must call build_head_model() first to initialize the branches.")

        input_ids = Input(shape=(self.max_len,), dtype=tf.int32, name="input_ids")
        attention_mask = Input(shape=(self.max_len,), dtype=tf.int32, name="attention_mask")
        token_type_ids = Input(shape=(self.max_len,), dtype=tf.int32, name="token_type_ids")

        out1 = self._branch_1([input_ids, attention_mask, token_type_ids])
        out2 = self._branch_2([input_ids, attention_mask, token_type_ids])

        distance = self.euclidean_distance(out1, out2)
        output = Dense(1, activation="sigmoid", name="similarity")(distance)

        return Model(
            inputs=[input_ids, attention_mask, token_type_ids],
            outputs=output
        )
