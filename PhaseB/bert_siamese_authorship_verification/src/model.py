import io
import tensorflow as tf
from transformers import TFBertModel
from contextlib import redirect_stdout
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Bidirectional, Dropout, LSTM, Lambda


class SiameseBertModel:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        self.bilstm_layers = self.config['model']['bilstm']['number_of_layers']
        self.bilstm_output_units = int(self.bilstm_layers * 2)
        self.bilstm_dropout = self.config['model']['bilstm']['dropout']

        self.filters = self.config['model']['cnn']['filters']
        self.pool_size = self.config['model']['cnn']['pool_size']
        self.padding = self.config['model']['cnn']['padding']
        self.kernel_size = self.config['model']['cnn']['kernel_size']

        self.in_features = self.config['model']['fc']['in_features']
        self.out_features = self.config['model']['fc']['out_features']

        self.bert_model = TFBertModel.from_pretrained('bert-base-uncased')
        self.bert_model.trainable = self.config['bert']['trainable']

        for var in self.bert_model.trainable_variables:
            print(var.name, var.shape)

        self.max_len = self.config['bert']['maximum_sequence_length']

        self._branch = None

    @staticmethod
    def get_model_summary_string(model):
        with io.StringIO() as buf, redirect_stdout(buf):
            model.summary()
            return buf.getvalue()

    def _build_siamese_branch(self):
        input_ids = Input(shape=(self.max_len,), dtype=tf.int32, name="input_ids")
        attention_mask = Input(shape=(self.max_len,), dtype=tf.int32, name="attention_mask")
        token_type_ids = Input(shape=(self.max_len,), dtype=tf.int32, name="token_type_ids")

        bert_output = self.bert_model({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        })[0]  # [0] is the last hidden state (full token embeddings)

        cnn_lstm_stack = Sequential(name="cnn_bilstm_stack")

        for i in range(len(self.kernel_size)):
            cnn_lstm_stack.add(Conv1D(filters=self.filters, kernel_size=self.kernel_size[i],
                                      padding=self.padding, activation='relu',
                                      input_shape=(self.max_len, 768)))
            cnn_lstm_stack.add(MaxPooling1D(pool_size=self.pool_size))

        cnn_lstm_stack.add(Bidirectional(
            LSTM(units=self.bilstm_output_units, return_sequences=True), merge_mode='concat'))
        cnn_lstm_stack.add(Bidirectional(
            LSTM(units=self.bilstm_output_units, go_backwards=True)
        ))

        cnn_lstm_stack.add(Dropout(self.bilstm_dropout))
        cnn_lstm_stack.add(Dense(self.in_features, activation='relu'))

        x = cnn_lstm_stack(bert_output)
        output = Dense(self.out_features, activation="relu", name="embedding")(x)

        return Model(inputs=[input_ids, attention_mask, token_type_ids], outputs=output)

    def build_model(self, model_name):
        self.logger.log(f"Started building model {model_name}...")

        # Shared branch
        self._branch = self._build_siamese_branch()
        branch_summary = self.get_model_summary_string(self._branch)
        self.logger.log(branch_summary)

        input_ids_1 = Input(shape=(self.max_len,), dtype=tf.int32, name="input_ids_1")
        attention_mask_1 = Input(shape=(self.max_len,), dtype=tf.int32, name="attention_mask_1")
        token_type_ids_1 = Input(shape=(self.max_len,), dtype=tf.int32, name="token_type_ids_1")

        input_ids_2 = Input(shape=(self.max_len,), dtype=tf.int32, name="input_ids_2")
        attention_mask_2 = Input(shape=(self.max_len,), dtype=tf.int32, name="attention_mask_2")
        token_type_ids_2 = Input(shape=(self.max_len,), dtype=tf.int32, name="token_type_ids_2")

        out1 = self._branch([input_ids_1, attention_mask_1, token_type_ids_1])
        out2 = self._branch([input_ids_2, attention_mask_2, token_type_ids_2])

        # Distance computation
        distance = Lambda(
            lambda tensors: tf.sqrt(tf.reduce_sum(tf.square(tensors[0] - tensors[1]), axis=1, keepdims=True) + 1e-6)
        )([out1, out2])

        # Output will be in [0, 1], suitable for BCE
        output = Dense(1, activation="sigmoid")(distance)

        model = Model(
            inputs=[
                input_ids_1, attention_mask_1, token_type_ids_1,
                input_ids_2, attention_mask_2, token_type_ids_2
            ],
            outputs=output
        )

        self.logger.log(f"Finished building model {model_name}...")
        return model

    def build_encoder_with_classifier(self):
        """ Creates a classifier model using the stored encoder branch """
        if self._branch is None:
            raise RuntimeError("You must call build_model() first to initialize the branch.")

        input_ids = Input(shape=self._branch.input[0].shape[1:], dtype=tf.int32, name="input_ids")
        attention_mask = Input(shape=self._branch.input[1].shape[1:], dtype=tf.int32, name="attention_mask")

        x = self._branch([input_ids, attention_mask])

        out = Dense(1, activation="sigmoid", name="chunk_classifier")(x)

        return Model(inputs=[input_ids, attention_mask], outputs=out)
