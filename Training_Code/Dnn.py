# Latent Space Deep Neural Network
# Ben Melanson
# May 20th, 2025

# Description
# This DNN allows for generating latent space data directly from the 
# airfoil profile data. This profile data is a combination of the
# coefficents used to generate the geometry along with the Mach Number,
# Angle of Attack, and Reynolds Number.


import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import activations
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import CSVLogger

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Uncomment this to disable GPU Processing. Probably a bad idea but good for debugging.

# Generates a Tensor for the Features
def featureProcessing(featurePaths):
    feature = tf.io.read_file(featurePaths)
    feature = tf.strings.split(feature, sep="\n") # Remove the new line characters.
    feature = feature[:featureCount] # Remove the trailing string characters.
    feature = tf.strings.to_number(feature) # Convert strings to floats.

    return feature

# Generates a Tensor for the X Labels
def labelProcessing(labelPaths):
    label = tf.io.read_file(labelPaths) # Get every file in the file path.
    label = tf.strings.split(label, sep="\n") # Seperate the floats.
    label = label[:-1]
    label = tf.strings.to_number(label)
    return label

def main(epochCount):

    featureDataFiles = tf.data.Dataset.list_files(f"{rootPath}{featureExtension}/*", shuffle = False) # Loads all the Features files into a Tf Dataset.
    labelDataFiles = tf.data.Dataset.list_files(f"{rootPath}{labelExtension}/*", shuffle = False) # Loads all the Label files into a Tf Dataset.

    labeledFeature = featureDataFiles.map(featureProcessing)
    labeledLabels = labelDataFiles.map(labelProcessing) # These just run the processing functions on the Datasets.

    dataset = tf.data.Dataset.zip(labeledFeature, labeledLabels)

    tf.data.Dataset.save(dataset, rootPath + "dnnDataset/") # Saves the entire dataset as a TensorFlow dataset object. Efficent for rapid saving and loading. REMOVE THIS IF SPACE IS AN ISSUE!!! (Or just use presaved datasets, im not your dad)
 
    dataset = tf.data.Dataset.load(rootPath + "dnnDataset/") # Loads the dataset object from storage. 

    num_samples = tf.data.experimental.cardinality(dataset).numpy()
    
    trainingSet = dataset.take(int(num_samples * 0.8))
    testingSet = dataset.skip(int(num_samples * 0.8))

    trainingSet = trainingSet.batch(100)
    testingSet = testingSet.batch(100)

    model = Sequential([
        Input([featureCount]),
        Dense(latentDim, activation = activations.relu),
        Dense(latentDim, activation = activations.relu),
        Dense(latentDim, activation = activations.relu),
        Dense(latentDim, activation = activations.relu),
        Dense(latentDim, activation = activations.relu),
        Dense(latentDim, activation = activations.relu),
    ])
    
    model.compile(
        optimizer = 'adam',
        loss = 'mse',
        metrics = ['accuracy'],
    )

    csvLogger = CSVLogger(str(f'{rootPath}denseModelTrainingHistory.csv'), append=True, separator=',')
    
    model.fit(
        trainingSet,
        epochs = epochCount,
        validation_data = testingSet,
        callbacks = [csvLogger] 
    )

    model.save(f"{rootPath}{dnnExt}")

if __name__ == "__main__":

    rootPath = os.path.dirname(__file__)
    rootPath = rootPath + '/'

    featureExtension = 'AirfoilFeatureData'
    labelExtension = 'AutoencoderLatentSpaceValues'

    imageShape = [84, 292]

    geometryFeatures = 14 # Geometry Modes, 14 for the Turbulent dataset and 30 for the Laminar Dataset
    additionalFeatures = 3 # Mach Number, Angle of Attack, and Reynolds Number. Dont change this value.

    featureCount = geometryFeatures + additionalFeatures

    latentDim = int((imageShape[0] * imageShape[1] * 0.25) + 2)

    dnnExt = 'dnn.keras'
    
    epochCount = 100
    
    main(epochCount)
  