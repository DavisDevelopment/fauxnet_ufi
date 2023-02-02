
import logging
import time

import numpy as np
import matplotlib.pyplot as plt

## import tensorflow_datasets as tfds
#import tensorflow as tf
#eras = tf.keras
import keras
#from keras import *
#from keras.layers import *


## encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
## encoder_l1 = tf.keras.layers.LSTM(100, return_state=True)
# encoder_outputs1 = encoder_l1(encoder_inputs)
# encoder_states1 = encoder_outputs1[1:]
## decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs1[0])
## decoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_inputs,initial_state = encoder_states1)
## decoder_outputs1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(decoder_l1)
## model_e1d1 = tf.keras.models.Model(encoder_inputs,decoder_outputs1)
# model_e1d1.summary()

class RecurrentAutoEncoder(Model):
   def __init__(self, n_past, n_features, n_future=1):
      super().__init__(name='recAE')
      self.n_past = n_past
      self.n_features = n_features
      self.n_future = n_future
      
      self.encoder = RecEncoder(n_past, n_features, n_future)
      self.decoder = RecDecoder(n_past, n_features, n_future)
      
   def call(self, inputs, training=None):
      enc_out, enc_states = self.encoder(inputs, training=training)
      return self.decoder(enc_out, enc_states, training=training)
   
class RecEncoder(Layer):
   def __init__(self, n_past, n_features, n_future=1, name=None):
      super().__init__(name=name)
      
      self.l1 = LSTM(100, return_state=True, unroll=True)
      self.l2 = Dense(800)
      
   def call(self, inputs, training=None):
      x = self.l1(inputs, training=training)
      states = x[1:]
      out = x[0]
      return self.l2(out, training=training), states
   
class RecDecoder(Layer):
   def __init__(self, n_past, n_features, n_future=1, name=None):
      super().__init__(name=name)
      self.l1 = LSTM(100, unroll=True)
      self.out = Dense(n_features)
      
   def call(self, inputs, encoder_states, training=None):
      x = self.l1(inputs, initial_state=encoder_states, training=training)
      return self.out(x, training=training)

#class MultiHeadAttention(tf.keras.layers.Layer):
   def __init__(self,*, d_model, num_heads):
      super(MultiHeadAttention, self).__init__()
      self.num_heads = num_heads
      self.d_model = d_model

      assert d_model % self.num_heads == 0

      self.depth = d_model // self.num_heads

#      self.wq = tf.keras.layers.Dense(d_model)
#      self.wk = tf.keras.layers.Dense(d_model)
#      self.wv = tf.keras.layers.Dense(d_model)

#      self.dense = tf.keras.layers.Dense(d_model)

   def split_heads(self, x, batch_size):
      """Split the last dimension into (num_heads, depth).
      Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
      """
#     x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
#     return tf.transpose(x, perm=[0, 2, 1, 3])

   def call(self, v, k, q, mask):
#     batch_size = tf.shape(q)[0]

      q = self.wq(q)  # (batch_size, seq_len, d_model)
      k = self.wk(k)  # (batch_size, seq_len, d_model)
      v = self.wv(v)  # (batch_size, seq_len, d_model)

      q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
      k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
      v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

      # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
      # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
      scaled_attention, attention_weights = scaled_dot_product_attention(
         q, k, v, mask)

#     scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

#     concat_attention = tf.reshape(scaled_attention,
                                    (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

      output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

      return output, attention_weights


#class EncoderLayer(tf.keras.layers.Layer):
   def __init__(self,*, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

#    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

#    self.dropout1 = tf.keras.layers.Dropout(rate)
#    self.dropout2 = tf.keras.layers.Dropout(rate)

   def call(self, x, training, mask):

    attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

    return out2
 

#class DecoderLayer(tf.keras.layers.Layer):
   def __init__(self,*, d_model, num_heads, dff, rate=0.1):
      super(DecoderLayer, self).__init__()

      self.mha1 = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
      self.mha2 = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

      self.ffn = point_wise_feed_forward_network(d_model, dff)

#      self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#      self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#      self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

#      self.dropout1 = tf.keras.layers.Dropout(rate)
#      self.dropout2 = tf.keras.layers.Dropout(rate)
#      self.dropout3 = tf.keras.layers.Dropout(rate)

   
   
   def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)

    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)

    attn2, attn_weights_block2 = self.mha2(
        enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

    return out3, attn_weights_block1, attn_weights_block2
 
#class Transformer(tf.keras.Model):
   def __init__(self,*, num_layers, d_model, num_heads, dff, input_vocab_size,
               target_vocab_size, rate=0.1):
      super().__init__()
      self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                              num_heads=num_heads, dff=dff,
                              input_vocab_size=input_vocab_size, rate=rate)

      self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                              num_heads=num_heads, dff=dff,
                              target_vocab_size=target_vocab_size, rate=rate)

#      self.final_layer = tf.keras.layers.Dense(target_vocab_size)

   def call(self, inputs, training):
#    # Keras models prefer if you pass all your inputs in the first argument
    inp, tar = inputs

    padding_mask, look_ahead_mask = self.create_masks(inp, tar)

    enc_output = self.encoder(inp, training, padding_mask)  # (batch_size, inp_seq_len, d_model)

    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    dec_output, attention_weights = self.decoder(
        tar, enc_output, training, look_ahead_mask, padding_mask)

    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

    return final_output, attention_weights

   def create_masks(self, inp, tar):
    # Encoder padding mask (Used in the 2nd attention block in the decoder too.)
    padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
#   look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
#   look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return padding_mask, look_ahead_mask

 
#?=======================[HELPER FUNCTIONS]=======================#

def create_padding_mask(seq):
#  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
   # add extra dimensions to add the padding
   # to the attention logits.
#  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
#  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
   return mask  # (seq_len, seq_len)

def scaled_dot_product_attention(q, k, v, mask):
   """
   -
    Calculate the attention weights.
      q, k, v must have matching leading dimensions.
      k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
      The mask has different shapes depending on its type(padding or look ahead)
      but it must be broadcastable for addition.

      Args:
         q: query shape == (..., seq_len_q, depth)
         k: key shape == (..., seq_len_k, depth)
         v: value shape == (..., seq_len_v, depth_v)
         mask: Float tensor with shape broadcastable
               to (..., seq_len_q, seq_len_k). Defaults to None.

      Returns:
         output, attention_weights
  """
#  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

   # scale matmul_qk
#  dk = tf.cast(tf.shape(k)[-1], tf.float32)
#  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

   # add the mask to the scaled tensor.
   if mask is not None:
      scaled_attention_logits += (mask * -1e9)

   # softmax is normalized on the last axis (seq_len_k) so that the scores
   # add up to 1.
#  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

#  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

   return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
#   return tf.keras.Sequential([
#      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
#      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
   ])