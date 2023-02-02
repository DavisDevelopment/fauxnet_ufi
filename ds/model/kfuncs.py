
import keras
#from tensorflow.python.ops import math_ops
##from tensorflow.keras.losses import *
#import tensorflow as tf
#import tensorflow.math as m
mo = math_ops

# @tf.function(jit_compile=False, autograph=False)
def moe(ytrue, ypred):
#  ytrue = tf.transpose(ytrue)
#  ypred = tf.transpose(ypred)
   
   d = mo.subtract(ytrue, ypred)
   # print(d)
   absd = mo.abs(d)
   # print(absd)
   error = m.divide(absd, ytrue)
   # print(error)
#  return tf.reduce_mean(tf.transpose(error))
