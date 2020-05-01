import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Dense, Dropout, Flatten, Input)
from tensorflow.keras.models import Model


def residual_block(x, filters_num):
    x_residual = x

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters_num, (3, 3), padding='same')(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters_num, (3, 3), padding='same')(x)

    x = tf.keras.layers.add([x, x_residual])

    return x


def downsample(x, filters_num):
    x = Conv2D(filters_num, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    return x


def resnet_simple_model():
    model_input = Input(shape=(150, 150, 3))
    x = BatchNormalization()(model_input)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(x)

    x = residual_block(x, 64)
    x = residual_block(x, 64)

    x = downsample(x, 128)

    x = residual_block(x, 128)
    x = residual_block(x, 128)

    x = downsample(x, 128)

    x = residual_block(x, 128)
    x = residual_block(x, 128)

    x = downsample(x, 64)

    x = Flatten(name='flatten')(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(3, activation='softmax')(x)

    return Model(inputs=model_input, outputs=x, name='simple_residual')
