import tensorflow as tf
from keras import Sequential
from tensorflow.keras import layers, Model
from transformers import TFBertModel
from config.get_config import get_config


def build_siamese_branch(bert_model):
    config = get_config()
    max_len = config['bert']['maximum_sequence_length']
    kernel_size = config['model']['cnn']['kernel_size']
    bilstm_layers = config['model']['bilstm']['hidden_units']
    filters = config['model']['cnn']['filters']
    stride = config['model']['cnn']['stride']
    out_features = config['model']['fc']['out_features']

    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")

    # 1. BERT Encoding
    bert_output = bert_model(input_ids, attention_mask=attention_mask)[0]  # (batch, seq_len, hidden_size)

    # 2. Post-BERT Processing in Sequential
    cnn_lstm_stack = Sequential(name="cnn_bilstm_stack")
    for i in range(len(kernel_size)):
        if i == 0:
            cnn_lstm_stack.add(layers.Conv1D(filters=filters, kernel_size=kernel_size[i], strides=stride,
                                             padding='valid', activation='relu',
                                             input_shape=(max_len, 768)))
        else:
            cnn_lstm_stack.add(layers.Conv1D(filters=filters, kernel_size=kernel_size[i], strides=stride,
                                             padding='valid', activation='relu'))
        cnn_lstm_stack.add(layers.MaxPooling1D(pool_size=2))  # pooling helps stabilize sequence length

    cnn_lstm_stack.add(layers.Bidirectional(layers.LSTM(
        units=bilstm_layers,
        return_sequences=True
    )))
    cnn_lstm_stack.add(layers.Bidirectional(layers.LSTM(
        units=bilstm_layers,
        go_backwards=True
    )))
    cnn_lstm_stack.add(layers.Dropout(0.25))
    cnn_lstm_stack.add(layers.Dense(128, activation='relu'))
    # cnn_lstm_stack.add(layers.Softmax(axis=config['model']['softmax_dim']))
    cnn_lstm_stack.add(layers.Dense(out_features, activation='relu'))

    # Apply CNN+BiLSTM Sequential block
    output = cnn_lstm_stack(bert_output)

    return Model(inputs=[input_ids, attention_mask], outputs=output)


def build_keras_siamese_model():
    config = get_config()
    bert_model = TFBertModel.from_pretrained(config['bert']['model'])

    # Two branches
    branch = build_siamese_branch(bert_model)

    input_ids1 = layers.Input(shape=(config['bert']['maximum_sequence_length'],), dtype=tf.int32, name="input_ids1")
    attention_mask1 = layers.Input(shape=(config['bert']['maximum_sequence_length'],), dtype=tf.int32,
                                   name="attention_mask1")

    input_ids2 = layers.Input(shape=(config['bert']['maximum_sequence_length'],), dtype=tf.int32, name="input_ids2")
    attention_mask2 = layers.Input(shape=(config['bert']['maximum_sequence_length'],), dtype=tf.int32,
                                   name="attention_mask2")

    out1 = branch([input_ids1, attention_mask1])
    out2 = branch([input_ids2, attention_mask2])

    # Compute L2 Euclidean distance between the outputs, with a small epsilon to avoid division by zero
    distance = layers.Lambda(
        lambda tensors: tf.sqrt(tf.reduce_sum(tf.square(tensors[0] - tensors[1]), axis=1, keepdims=True) + 1e-6))(
        [out1, out2])

    # Final sigmoid for binary classification
    output = layers.Dense(1, activation='sigmoid')(distance)

    return Model(inputs=[input_ids1, attention_mask1, input_ids2, attention_mask2], outputs=output)
