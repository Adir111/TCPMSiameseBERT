import tensorflow as tf
from tensorflow.keras import layers, Model
from transformers import TFBertModel
from config.get_config import get_config


def build_siamese_branch(bert_model):
    config = get_config()
    input_ids = layers.Input(shape=(config['bert']['maximum_sequence_length'],), dtype=tf.int32, name="input_ids")
    attention_mask = layers.Input(shape=(config['bert']['maximum_sequence_length'],), dtype=tf.int32,
                                  name="attention_mask")

    bert_output = bert_model(input_ids, attention_mask=attention_mask)[0]  # last_hidden_state

    # Mean pooling
    mask_expanded = tf.cast(tf.expand_dims(attention_mask, -1), tf.float32)
    sum_embeddings = tf.reduce_sum(bert_output * mask_expanded, axis=1)
    sum_mask = tf.reduce_sum(mask_expanded, axis=1)
    pooled = sum_embeddings / tf.clip_by_value(sum_mask, 1e-9, 1e9)

    # CNN expects (batch, channels, seq_len) → we simulate with Conv1D over features
    cnn_input = tf.expand_dims(pooled, axis=-1)  # (batch, features, 1)
    cnn_input = tf.transpose(cnn_input, perm=[0, 2, 1])  # (batch, 1, features)

    conv = layers.Conv1D(
        filters=config['model']['cnn']['filters'],
        kernel_size=config['model']['cnn']['kernel_size'],
        strides=config['model']['cnn']['stride'],
        padding='same'
    )(cnn_input)

    pool = layers.GlobalMaxPooling1D()(conv)

    lstm = layers.Reshape((1, -1))(pool)
    lstm = layers.Bidirectional(layers.LSTM(
        units=config['model']['bilstm']['hidden_units'],
        dropout=config['model']['bilstm']['dropout'],
        return_sequences=False
    ))(lstm)

    dense = layers.Dense(128, activation='relu')(lstm)
    softmax_out = layers.Softmax(axis=config['model']['softmax_dim'])(dense)

    final = layers.Dense(config['model']['fc']['out_features'], activation='relu')(softmax_out)
    final = layers.Dense(1)(final)

    return Model(inputs=[input_ids, attention_mask], outputs=final)


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

    # Compute L1 distance between the outputs
    distance = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([out1, out2])

    return Model(inputs=[input_ids1, attention_mask1, input_ids2, attention_mask2], outputs=distance)
