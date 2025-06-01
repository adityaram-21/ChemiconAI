import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
import numpy as np

class PositionalEncoding(layers.Layer):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        pos_encoding = self._positional_encoding(max_seq_len, d_model)
        self.pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)

    def _get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def _positional_encoding(self, position, d_model):
        pos = np.arange(position)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rads = self._get_angles(pos, i, d_model)

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])  # even indices
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])  # odd indices

        return angle_rads[np.newaxis, ...]

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

class ImageToSmilesModel:
    def __init__(self, vocab_size, max_seq_len, d_model=512, num_heads=4, dff=1024):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff

        self.model = self._build_model()

    def _build_encoder(self):
        base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
        base_model.trainable = False

        model = tf.keras.Sequential([
            layers.Conv2D(3, (3, 3), padding='same', input_shape=(256, 256, 1)),  # Convert grayscale to 3-channels
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(self.d_model)  # Final feature vector
        ])

        return model

    def _build_decoder(self):
        inputs = layers.Input(shape=(None,))  # SMILES token sequence
        enc_output = layers.Input(shape=(self.d_model,))  # Encoder output

        x = layers.Embedding(self.vocab_size, self.d_model)(inputs)
        x = PositionalEncoding(self.max_seq_len, self.d_model)(x)

        # Self-attention
        attn1 = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.d_model)(x, x)
        x = layers.LayerNormalization()(x + attn1)

        # Encoder-decoder attention
        enc_output_exp = layers.Reshape((1, self.d_model))(enc_output)
        attn2 = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.d_model)(x, enc_output_exp)
        x = layers.LayerNormalization()(x + attn2)

        # Feed forward network
        ffn = layers.Dense(self.dff, activation='relu')(x)
        ffn = layers.Dense(self.d_model)(ffn)
        x = layers.LayerNormalization()(x + ffn)

        outputs = layers.Dense(self.vocab_size, activation='softmax')(x)

        return tf.keras.Model([inputs, enc_output], outputs)

    def _build_model(self):
        img_input = layers.Input(shape=(256, 256, 1))
        seq_input = layers.Input(shape=(None,))

        encoder = self._build_encoder()
        decoder = self._build_decoder()

        enc_output = encoder(img_input)
        dec_output = decoder([seq_input, enc_output])

        return tf.keras.Model(inputs=[img_input, seq_input], outputs=dec_output)

    def summary(self):
        self.model.summary()

    def get_model(self):
        return self.model