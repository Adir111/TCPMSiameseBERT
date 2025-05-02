import numpy as np
import tensorflow as tf
from transformers import TFBertModel
from PhaseB.bert_siamese_authorship_verification.utilities.env_handler import is_tf_2_10
from siamese import SiameseNetwork

if is_tf_2_10():
    from keras import Input, Model
    from keras.layers import Dense, Bidirectional, Dropout, LSTM, Lambda
    from keras.layers.convolutional import Conv1D, MaxPooling1D
else:
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Bidirectional, Dropout, LSTM, Lambda


class SiameseBertModel:
    def __init__(self, config):
        self.kernel_size = config['model']['cnn']['kernel_size']
        bilstm_layers = config['model']['bilstm']['number_of_layers']
        self.bilstm_output_units = int(bilstm_layers * 2)
        self.bilstm_dropout = config['model']['bilstm']['dropout']
        self.filters = config['model']['cnn']['filters']
        self.pool_size = config['model']['cnn']['pool_size']
        self.in_features = config['model']['fc']['in_features']
        self.out_features = config['model']['fc']['out_features']
        self.padding = config['model']['cnn']['padding']
        self.max_len = config['bert']['maximum_sequence_length']
        self.hidden_size = config['training']['hidden_size']
        self.bert_model = TFBertModel.from_pretrained(config['bert']['model'])
        self.bert_model.trainable = config['bert']['trainable']

    def __get_cnn_bilstm_stack(self, input_shape):
        inputs = Input(shape=input_shape)

        # CNN Layer
        x = Conv1D(filters=self.filters, kernel_size=self.kernel_size, padding=self.padding)(inputs)
        x = MaxPooling1D(pool_size=self.pool_size)(x)

        # BiLSTM Layer
        x = Bidirectional(LSTM(self.bilstm_output_units, return_sequences=False))(x)

        # Fully Connected Layer
        x = Dropout(self.bilstm_dropout)(x)
        x = Dense(self.in_features, activation='relu')(x)  # For binary classification

        # Output Layer
        outputs = Dense(self.out_features, activation='relu')(x)
        model = Model(inputs=inputs, outputs=outputs, name="cnn_bilstm_stack")
        return model

    def __create_base_model(self):
        combined_input = Input(shape=(self.max_len * 2,), dtype=np.int32, name="combined_input")
        # input_ids = Input(shape=self.input_shape, dtype=tf.int32, name="input_ids")
        # attention_mask = Input(shape=self.input_shape, dtype=tf.int32, name="attention_mask")
        input_ids = Lambda(lambda x: tf.convert_to_tensor(x[:, :self.max_len], dtype=tf.int32))(combined_input)
        attention_mask = Lambda(lambda x: tf.convert_to_tensor(x[:, self.max_len:], dtype=tf.int32))(combined_input)

        # BERT output
        bert_output = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )[0]

        # CNN BiLSTM output
        cnn_bilstm_stack = self.__get_cnn_bilstm_stack((self.max_len, self.hidden_size))
        output = cnn_bilstm_stack(bert_output)

        return Model(inputs=combined_input, outputs=output, name="siamese_branch")

    @staticmethod
    def __create_head_model(embedding_shape):
        embedding_a = Input(shape=embedding_shape)
        embedding_b = Input(shape=embedding_shape)

        # Euclidean distance
        distance = Lambda(
            lambda tensors: tf.sqrt(tf.reduce_sum(tf.square(tensors[0] - tensors[1]), axis=1, keepdims=True) + 1e-6),
            name="euclidean_distance"
        )([embedding_a, embedding_b])

        # Output layer
        output = Dense(1, activation="sigmoid", name="similarity_score")(distance)

        return Model(inputs=[embedding_a, embedding_b], outputs=output, name="siamese_head_model")

    def build_model(self):
        base_model = self.__create_base_model()
        head_model = self.__create_head_model(base_model.output_shape[1:])
        siamese_network = SiameseNetwork(base_model, head_model)
        return siamese_network