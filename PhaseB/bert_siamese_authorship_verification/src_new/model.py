import tensorflow as tf
from transformers import TFBertModel

from PhaseB.bert_siamese_authorship_verification.utilities.env_handler import is_tf_2_10

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

    def __get_cnn_bilstm_stack(self):
        inputs = Input(shape=(self.max_len, self.hidden_size))

        # CNN Layer
        x = Conv1D(filters=self.filters, kernel_size=self.kernel_size, padding=self.padding)(inputs)
        x = MaxPooling1D(pool_size=self.pool_size)(x)

        # BiLSTM Layer
        x = Bidirectional(LSTM(self.bilstm_output_units, return_sequences=False))(x)

        # Fully Connected Layer
        x = Dropout(self.bilstm_dropout)(x)
        x = Dense(self.in_features, activation='relu')(x) # For binary classification

        # Output Layer
        outputs = Dense(self.out_features, activation='relu')(x)
        model = Model(inputs=inputs, outputs=outputs, name="cnn_bilstm_stack")
        return model

    def __build_siamese_branch(self):
        input_ids = Input(shape=(self.max_len,), dtype=tf.int32, name="input_ids")
        attention_mask = Input(shape=(self.max_len,), dtype=tf.int32, name="attention_mask")

        # BERT output
        bert_output = self.bert_model(input_ids, attention_mask=attention_mask)[0]
        print(f"BERT output shape: {bert_output.shape}")  # Debug print

        # CNN BiLSTM output
        cnn_bilstm_stack = self.__get_cnn_bilstm_stack()
        output = cnn_bilstm_stack(bert_output)
        print(f"Output after CNN+BiLSTM: {output.shape}")  # Debug print

        return Model(inputs=[input_ids, attention_mask], outputs=output, name="siamese_branch")

    def build_model(self):
        """
        Build the full model with BERT + CNN + BiLSTM layers.
        """
        # Shared branch
        shared_branch = self.__build_siamese_branch()

        # Define the pair inputs for BERT
        input_ids_1 = Input(shape=(self.max_len,), dtype=tf.int32, name="input_text_1")
        attention_mask_1 = Input(shape=(self.max_len,), dtype=tf.int32, name="attention_mask_1")
        input_ids_2 = Input(shape=(self.max_len,), dtype=tf.int32, name="input_text_2")
        attention_mask_2 = Input(shape=(self.max_len,), dtype=tf.int32, name="attention_mask_2")

        # Get BERT embeddings for both inputs
        bert_output_1 = shared_branch([input_ids_1, attention_mask_1])
        bert_output_2 = shared_branch([input_ids_2, attention_mask_2])

        # Distance computation
        distance = Lambda(
            lambda tensors: tf.sqrt(tf.reduce_sum(tf.square(tensors[0] - tensors[1]), axis=1, keepdims=True) + 1e-6)
        )([bert_output_1, bert_output_2])

        # Output will be in [0, 1], suitable for BCE
        output = Dense(1, activation="sigmoid")(distance)

        # Final Model
        model = Model(inputs=[input_ids_1, attention_mask_1, input_ids_2, attention_mask_2], outputs=output, name="bert_siamese_with_cnn_bilstm")
        return model