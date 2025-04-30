import io
import tensorflow as tf
from transformers import TFBertModel
from contextlib import redirect_stdout

from PhaseB.bert_siamese_authorship_verification.utilities.env_handler import is_tf_2_10

if is_tf_2_10():
    from keras import Sequential, Input, Model
    from keras.layers import Dense, Bidirectional, Dropout, LSTM, Lambda
    from keras.layers.convolutional import Conv1D, MaxPooling1D
else:
    from tensorflow.keras import Sequential, Input, Model
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Bidirectional, Dropout, LSTM, Lambda

class SiameseBertModel:
    def __init__(self, config, logger, model_name="Default"):
        self.config = config
        self.logger = logger
        self.kernel_size = self.config['model']['cnn']['kernel_size']
        self.bilstm_layers = self.config['model']['bilstm']['number_of_layers']
        self.bilstm_output_units = int(self.bilstm_layers * 2)
        self.bilstm_dropout = self.config['model']['bilstm']['dropout']
        self.filters = self.config['model']['cnn']['filters']
        self.pool_size = self.config['model']['cnn']['pool_size']
        self.in_features = self.config['model']['fc']['in_features']
        self.out_features = self.config['model']['fc']['out_features']
        self.padding = self.config['model']['cnn']['padding']
        self.max_len = self.config['bert']['maximum_sequence_length']
        self.hidden_size = config['training']['hidden_size']

        self.bert_model = TFBertModel.from_pretrained(self.config['bert']['model'])
        self.bert_model.trainable = self.config['bert']['trainable']

        self._branch = None
        self._model_name = model_name

    @staticmethod
    def __mean_pooling(last_hidden_state, attention_mask):
        """
        Perform mean pooling on the output of BERT's last hidden state.
        """
        # Mask padding tokens (attention_mask is 1 for real tokens, 0 for padding)
        mask_expanded = tf.expand_dims(attention_mask, axis=-1)  # Shape: (batch_size, seq_len, 1)
        masked_last_hidden_state = last_hidden_state * tf.cast(mask_expanded, tf.float32) # Apply the mask to the last_hidden_state
        sum_hidden_state = tf.reduce_sum(masked_last_hidden_state, axis=1) # Sum the hidden states across the sequence length (seq_len) dimension, where padding is ignored
        real_token_count = tf.reduce_sum(mask_expanded, axis=1) # Count the number of real tokens (excluding padding)
        real_token_count = tf.cast(real_token_count, tf.float32)
        mean_pooled_output = sum_hidden_state / real_token_count # Compute the mean by dividing the sum by the number of real tokens
        return mean_pooled_output

    def __run_bert_model(self, preprocessed_collection):
        """
        Pass the tokenized inputs into the BERT model and perform mean pooling.
        """
        pooled_outputs = []

        for tokenized_input in preprocessed_collection:
            # Extract the tokenized input (input_ids, attention_mask) from the dictionary
            input_ids = tf.convert_to_tensor(tokenized_input['input_ids'])
            attention_mask = tf.convert_to_tensor(tokenized_input['attention_mask'])

            inputs = { 'input_ids': input_ids, 'attention_mask': attention_mask } # BERT expects inputs as a dictionary
            output = self.bert_model(inputs) # Forward pass through the BERT model
            last_hidden_state = output.last_hidden_state # Get the last hidden state (shape: [batch_size, seq_len, hidden_size])
            pooled_output = self.__mean_pooling(last_hidden_state, attention_mask) # Apply mean pooling
            pooled_outputs.append(pooled_output)

        pooled_outputs = tf.stack(pooled_outputs) # Stack the pooled outputs to create a tensor of shape [batch_size, hidden_size]
        return pooled_outputs

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

    def build_complete_model(self):
        """
        Build the full model with BERT preprocessing + CNN + BiLSTM layers.
        """
        # Define the input for the BERT preprocessing
        input_ids_1 = Input(shape=(self.max_len,), dtype=tf.int32, name="input_text1")
        attention_mask_1 = Input(shape=(self.max_len,), dtype=tf.int32, name="attention_mask1")
        input_ids_2 = Input(shape=(self.max_len,), dtype=tf.int32, name="input_text2")
        attention_mask_2 = Input(shape=(self.max_len,), dtype=tf.int32, name="attention_mask2")

        # Get BERT embeddings for both inputs
        bert_output_1 = self.bert_model(input_ids=input_ids_1, attention_mask=attention_mask_1)[1]  # shape: (None, 768)
        bert_output_2 = self.bert_model(input_ids=input_ids_2, attention_mask=attention_mask_2)[1]  # shape: (None, 768)

        # Shared CNN + BiLSTM branch
        cnn_bilstm_model = self.__get_cnn_bilstm_stack()
        out1 = cnn_bilstm_model(bert_output_1)
        out2 = cnn_bilstm_model(bert_output_2)

        # Compute Euclidean Distance
        distance = Lambda(lambda tensors: tf.sqrt(tf.reduce_sum(tf.square(tensors[0] - tensors[1]), axis=1, keepdims=True)))([out1, out2])

        # Final Model
        model = Model(inputs=[input_ids_1, attention_mask_1, input_ids_2, attention_mask_2], outputs=distance)
        return model