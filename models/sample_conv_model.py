"""
Example model file, this file holds some functions to build a model with a
specific architecture, a convolution model in this case
"""

import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten


def basic_conv_model(input_shape):
    input_layer = Input(shape=input_shape)
    y = Conv2D(num_kernels[0], kernel_sizes[0], activation='relu')(input_layer)
    y = MaxPooling2D(pool_sizes[0], pool_strides[0])(y)
    y = Conv2D(num_kernels[1], kernel_sizes[1], activation='relu')(y)
    y = MaxPooling2D(pool_sizes[1], pool_strides[1])(y)
    y = Flatten()(y)
    y = Dense(num_hidden_units, activation='relu')(y)
    output_layer = Dense(num_classes, activation='softmax')(y)
    return input_layer, output_layer
