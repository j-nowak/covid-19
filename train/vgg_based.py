from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,
                                     Dropout, Flatten, Input)
from tensorflow.keras.models import Model


def vgg16_based_model():
  model_input = Input(shape=(150, 150, 3))

  base_model = VGG16(weights='imagenet', include_top=False, input_tensor=model_input)
  for layer in base_model.layers:
    layer.trainable = False

  x = base_model.output
  x = Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
  x = BatchNormalization()(x)

  x = Flatten(name='flatten')(x)
  x = Dense(64, activation='relu')(x)
  x = BatchNormalization()(x)
  x = Dropout(0.5)(x)
  x = Dense(3, activation='softmax')(x)

  return Model(inputs=base_model.input, outputs=x, name='vgg16_based')
