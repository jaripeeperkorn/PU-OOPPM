# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 17:44:38 2022

@author: orteg
"""
import math
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_array, check_consistent_length

from sklearn.utils import check_random_state
import tensorflow as tf
from tensorflow.keras.backend import epsilon
import tensorflow.keras.backend as K

def binary_crossentropy_(target, output, from_logits=False):
    if from_logits:
        output = K.sigmoid(output)
    output = K.clip(output, epsilon(), 1.0 - epsilon())
    output = -target * K.log(output) - (1.0 - target) * K.log(1.0 - output)
    return output


def uPU_loss(label_freq):
    
  def pu_logloss(y_true, y_pred, from_logits=False, label_smoothing=0):
      y_pred = K.constant(y_pred) if not tf.is_tensor(y_pred) else y_pred
      y_true = K.cast(y_true, y_pred.dtype)
      if label_smoothing is not 0:
          smoothing = K.cast_to_floatx(label_smoothing)
          y_true = K.switch(K.greater(smoothing, 0),
                            lambda: y_true * (1.0 - smoothing) + 0.5 * smoothing,
                            lambda: y_true)
      return K.mean(pu_logloss_(y_true, y_pred, label_freq, from_logits=from_logits), axis=-1)
  return pu_logloss
  
  
def pu_logloss_(target, output, label_freq, from_logits=False):
    if from_logits:
        output = K.sigmoid(output)
    output = K.clip(output, epsilon(), 1.0 - epsilon())
    output = target*( K.log(output)/label_freq + (1.0-1.0/label_freq) * K.log(1.0-output) ) + (1.0-target) * K.log(1.0-output)
    return -output
  
def nnPU_loss(label_freq):
    
  def nnpu_logloss(y_true, y_pred, from_logits=False, label_smoothing=0):
      y_pred = K.constant(y_pred) if not tf.is_tensor(y_pred) else y_pred
      y_true = K.cast(y_true, y_pred.dtype)
      if label_smoothing is not 0:
          smoothing = K.cast_to_floatx(label_smoothing)
          y_true = K.switch(K.greater(smoothing, 0),
                            lambda: y_true * (1.0 - smoothing) + 0.5 * smoothing,
                            lambda: y_true)
      return K.mean(nnpu_logloss_(y_true, y_pred, label_freq, from_logits=from_logits), axis=-1)
  return nnpu_logloss
  
  
def nnpu_logloss_(target, output, label_freq, from_logits=False):
    if from_logits:
        output = K.sigmoid(output)
    output = K.clip(output, epsilon(), 1.0 - epsilon())
    max_tensor = K.maximum(0.0, -1*( (1.0-target)*K.log(1.0-output)+target*(1.0-1.0/label_freq)*K.log(1.0-output) ) )
    output = -target*(K.log(output)/label_freq) + max_tensor                      
    return output

                      
                      
