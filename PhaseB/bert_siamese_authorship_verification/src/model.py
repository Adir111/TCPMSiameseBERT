import io
import tensorflow as tf
from keras import Sequential, Input, Model
from keras.layers import Dense, Bidirectional, Dropout, LSTM, Lambda
from keras.layers.convolutional import MaxPooling1D, Conv1D
from transformers import TFBertModel
from contextlib import redirect_stdout


class SiameseBertModel:
    def __init__(self, config, logger, model_name="Default"):
        self.config = config
        self.logger = logger
        self.kernel_size = self.config['model']['cnn']['kernel_size']
        self.bilstm_layers = self.config['model']['bilstm']['number_of_layers']
        self.bilstm_output_units = int(self.bilstm_layers * 2)
        self.bilstm_dropout = self.config['model']['bilstm']['dropout']
        self.embedding_dim = self.config['model']['bilstm']['dimension']
        self.filters = self.config['model']['cnn']['filters']
        self.stride = self.config['model']['cnn']['stride']
        self.pool_size = self.config['model']['cnn']['pool_size']
        self.in_features = self.config['model']['fc']['in_features']
        self.out_features = self.config['model']['fc']['out_features']
        self.padding = self.config['model']['cnn']['padding']

        self.chunk_size = self.config['training']['batch_size'] // self.config['training']['chunk_factor']

        self.bert_model = TFBertModel.from_pretrained(self.config['bert']['model'])
        self.bert_model.trainable = self.config['bert']['trainable']

        self._branch = None
        self._model_name = model_name

    def get_model_name(self):
        return self._model_name

    def set_model_name(self, name):
        self._model_name = name

    @staticmethod
    def get_model_summary_string(model):
        with io.StringIO() as buf, redirect_stdout(buf):
            model.summary()
            return buf.getvalue()

    def _build_siamese_branch(self):
        input_ids = Input(shape=(self.chunk_size,), dtype=tf.int32, name="input_ids")
        attention_mask = Input(shape=(self.chunk_size,), dtype=tf.int32, name="attention_mask")

        # BERT output
        bert_output = self.bert_model(input_ids, attention_mask=attention_mask)[0]

        # CNN + BiLSTM Stack
        cnn_lstm_stack = Sequential(name="cnn_bilstm_stack")

        for i in range(len(self.kernel_size)):
            cnn_lstm_stack.add(Conv1D(filters=self.filters, kernel_size=self.kernel_size[i],
                                      padding=self.padding, activation='relu',
                                      input_shape=(self.chunk_size, self.embedding_dim)))
            cnn_lstm_stack.add(MaxPooling1D(pool_size=self.pool_size))

        cnn_lstm_stack.add(Bidirectional(
            LSTM(units=self.bilstm_output_units, return_sequences=True, merge_mode='concat')))
        cnn_lstm_stack.add(Bidirectional(
            LSTM(units=self.bilstm_output_units, go_backwards=True)
        ))

        cnn_lstm_stack.add(Dropout(self.bilstm_dropout))
        cnn_lstm_stack.add(Dense(self.in_features, activation='relu'))
        cnn_lstm_stack.add(Dense(self.out_features, activation='relu'))

        output = cnn_lstm_stack(bert_output)

        return Model(inputs=[input_ids, attention_mask], outputs=output)

    def build_model(self):
        self.logger.log(f"Started building model {self.get_model_name()}...")

        # Shared branch
        self._branch = self._build_siamese_branch()

        # Inputs for pair 1
        input_ids1 = Input(shape=(self.chunk_size,), dtype=tf.int32, name="input_ids1")
        attention_mask1 = Input(shape=(self.chunk_size,), dtype=tf.int32, name="attention_mask1")

        # Inputs for pair 2
        input_ids2 = Input(shape=(self.chunk_size,), dtype=tf.int32, name="input_ids2")
        attention_mask2 = Input(shape=(self.chunk_size,), dtype=tf.int32, name="attention_mask2")

        # Branch outputs
        out1 = self._branch([input_ids1, attention_mask1])
        out2 = self._branch([input_ids2, attention_mask2])

        # Distance computation
        distance = tf.abs(Lambda(
            lambda tensors: tf.sqrt(tf.reduce_sum(tf.square(tensors[0] - tensors[1]), axis=1, keepdims=True) + 1e-6)
        )([out1, out2]))

        model = Model(inputs=[input_ids1, attention_mask1, input_ids2, attention_mask2], outputs=distance)
        self.logger.log(f"Finished building model {self.get_model_name()}...")
        return model

    def build_encoder_with_classifier(self):
        """ Creates a classifier model using the stored encoder branch """
        if self._branch is None:
            raise RuntimeError("You must call build_model() first to initialize the branch.")

        input_ids = tf.keras.Input(shape=self._branch.input[0].shape[1:], dtype=tf.int32, name="input_ids")
        attention_mask = tf.keras.Input(shape=self._branch.input[1].shape[1:], dtype=tf.int32, name="attention_mask")

        x = self._branch([input_ids, attention_mask])

        # Todo: replace with Relu activation
        out = tf.keras.layers.Dense(1, activation="sigmoid", name="chunk_classifier")(x)

        return tf.keras.Model(inputs=[input_ids, attention_mask], outputs=out)
