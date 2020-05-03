from datetime import datetime
from simple_residual import resnet_simple_model
from vgg_based import vgg16_based_model
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_TRAIN = './data/processed/train'
DATASET_TEST = './data/processed/test'
SAVED_MODELS_PATH = './saved_models'

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', help='vgg16_based or simple_residual')
args = parser.parse_args()


def data_generators(batch_size=16):
  train_datagen = ImageDataGenerator(
          rescale=1./255,
          shear_range=0.2,
          zoom_range=0.2,
          horizontal_flip=True)
  train_generator = train_datagen.flow_from_directory(
          DATASET_TRAIN,
          target_size=(150, 150),
          batch_size=batch_size,
          class_mode='categorical',
          shuffle=True)

  test_datagen = ImageDataGenerator(rescale=1./255)
  validation_generator = test_datagen.flow_from_directory(
          DATASET_TEST,
          target_size=(150, 150),
          batch_size=batch_size,
          class_mode='categorical',
          shuffle=False)

  return train_generator, validation_generator


def build_model(model_type):
  if model_type == 'vgg16_based':
    return vgg16_based_model()
  elif model_type == 'simple_residual':
    return resnet_simple_model()
  else:
    raise Exception('Model type not known')


def model_dir_path(model_name):
  time_now = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
  return SAVED_MODELS_PATH + '/' + model_name + '_' + time_now


def fit_model(model, train_generator, validation_generator):
  step_size_train=train_generator.n//train_generator.batch_size
  step_size_valid=validation_generator.n//validation_generator.batch_size

  path_to_save = model_dir_path(model.name)
  cp_callback = tf.keras.callbacks.ModelCheckpoint(path_to_save, save_best_only=True)

  model.fit_generator(
        train_generator,
        steps_per_epoch=step_size_train,
        epochs=60,
        validation_data=validation_generator,
        validation_steps=step_size_valid,
        callbacks=[cp_callback])


model = build_model(args.model_type)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

train_generator, validation_generator = data_generators()
fit_model(model, train_generator, validation_generator)