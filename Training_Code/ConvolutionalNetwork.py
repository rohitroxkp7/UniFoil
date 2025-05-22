# Convolutional Encoding Network
# Ben Melanson
# May 21st, 2025

# Description
# This script contains the Convolutional Neural Network. It is used to
# go between an encoded latent space and the pressure field data.

import os
import numpy as np # Needed for debugging purposes, do not remove
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import CSVLogger

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Uncomment this to disable GPU Processing. Probably a bad idea but good for debugging.

# Generates a Tensor for the Features
def featureProcessing(featurePaths):
    feature = tf.io.read_file(featurePaths)
    feature = tf.strings.split(feature, sep="\n") # Remove the new line characters.
    feature = feature[:-1] # Remove the trailing string characters.
    feature = tf.strings.to_number(feature) # Convert strings to floats.
    return feature

# Generates a Tensor for the X Labels
def labelProcessing(labelPaths):
    label = tf.io.read_file(labelPaths) # Get every file in the file path.
    label = tf.strings.split(label, sep="\n") # Remove the new line and row characters.
    label = label[:-1] # Remove the trailing string characters.
    label = tf.strings.split(label, sep=" ") # Seperate all numbers to a long list.
    label = tf.strings.to_number(label) # Convert strings to floats.
    label = tf.reshape(label, imageShape)

    return label

def L2Norm(y_true, y_pred):
  # This takes the L2 Norm of the provided two matricies.

  y_true = tf.reshape(y_true, [-1, imageShape[0], imageShape[1]])
  y_pred = tf.reshape(y_pred, [-1, imageShape[0], imageShape[1]])

  norm = tf.norm(y_true - y_pred) / tf.norm(y_true)
  return norm

class AutoEncode(Model):
  def __init__(self):
    super(AutoEncode, self).__init__()
    # The encoder model, converts flow field values into the latent space.
    self.encoder = tf.keras.Sequential([
      layers.Input(shape = (imageShape[0], imageShape[1], 1)),
      layers.MaxPool2D((2, 2)),
      layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=1),
      layers.MaxPool2D((2, 2)),
      layers.Conv2D(4, (3, 3), activation='relu', padding='same', strides=1),
      layers.Flatten(),
      layers.Dense(int(imageShape[0] * imageShape[1] * (0.25)), activation = 'relu'),
    ])

    # The Decoder Model, uses upsampling to return latent space values to flow field values.
    self.decoder = tf.keras.Sequential([
      layers.Dense(int(imageShape[0] * imageShape[1] * (0.25)), activation = 'relu'),
      layers.Reshape([int(imageShape[0] / 4), int(imageShape[1] / 4), 4]),
      layers.Conv2DTranspose(32, kernel_size=3, strides=1, activation='relu', padding='same'),
      layers.UpSampling2D((2,2)),
      layers.Conv2DTranspose(8, kernel_size=3, strides=1, activation='relu', padding='same'),
      layers.UpSampling2D((2,2)),
      layers.Conv2DTranspose(1, kernel_size=3, strides=1, activation='relu', padding='same'),
    ])

  def call(self, x):
    # This just makes sure that any requests to use the model runs through both the encoding and decoding steps.
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

def main(epochCount):

  labelDataFiles = tf.data.Dataset.list_files(f"{rootPath}{labelExtension}/*", shuffle = False) # Loads all the Label files into a Tf Dataset.
  labeledLabels = labelDataFiles.map(labelProcessing) # These just run the processing functions on the Datasets.

  dataset = tf.data.Dataset.zip(labeledLabels, labeledLabels) # Merges the datasets for features and labels into one.
  
  """
  currentVal = 0
  for unit in tf.data.Dataset.as_numpy_iterator(dataset.take(100)): # Subsection used to test the feature and label compilation. If input data format changes use this to check.
      currentVal = currentVal + 1
      if np.isnan(unit[0]).any():
         print(unit)
         print(currentVal)
         print(labelDataFiles[currentVal - 1])
         print("Is nan")
      if np.shape(unit[0]) != (imageShape[0], imageShape[1]):
         print("Bad read")
  """

  tf.data.Dataset.save(dataset, rootPath + "encoderDataset/") # Saves the compiled dataset as shown. Takes up more disk space but saves on CPU resources. Feel free to comment out.

  dataset = tf.data.Dataset.load(rootPath + "encoderDataset/") # Loads the dataset as saved in the above line.

  num_samples = tf.data.experimental.cardinality(dataset).numpy()

  trainingSet = dataset.take(int(num_samples * 0.8)) # Trains the model on the first 80% of the dataset. This might yield worse results then using a shuffler.
  testingSet = dataset.skip(int(num_samples * 0.8)) # Keeps the last 20% of the dataset for validation

  trainingSet = trainingSet.batch(32) # Splits the dataset into batches of 32 for training. Increase this if you have more VRAM, decrease it if you have less.
  validationSet = testingSet.batch(32)

  autoencoder = AutoEncode() # Defines the autoencoder object.

  autoencoder.compile(
    # This defines the autoencoder as a TensorFlow Model.
    optimizer = 'adam',
    loss = 'mse',
    metrics = ['accuracy', L2Norm]
  )

  autoencoder.encoder.summary() # Some debug printouts to show that it is working :)
  autoencoder.decoder.summary()

  csvLogger = CSVLogger(str(f'{rootPath}encodingModelTrainingHistory.csv'), append=True, separator=',') # Logs the epoch data to a .csv. Make sure to delete the .csv after each training run!

  autoencoder.fit(
    # This runs the training for the encoder and decoder.
    trainingSet,
    epochs = epochCount,
    validation_data = validationSet,
    callbacks = [csvLogger]
    )

  autoencoder.encoder.save(str(f"{rootPath}{encoderExt}")) # Saves the encoding model to the encoderExt filename
  
  autoencoder.decoder.save(str(f"{rootPath}{decoderExt}")) # Saves the decoding model to the decoderExt filename

if __name__ == "__main__":

  rootPath = os.path.dirname(__file__)
  rootPath = rootPath + '/'

  labelExtension = 'AirfoilPressureData'
  encoderExt = 'encoder.keras'
  decoderExt = 'deceoder.keras'
    
  epochCount = 100 # The amount of epochs run by default.

  imageShape = [84, 292] # The shape of the pressure data.
    
  main(epochCount)
  
