# Encoder
# Ben Melanson
# May 17th, 2025

# Description
# This script was designed as a backup but now serves
# as the primary method for encoding the pressure data
# into the latent space. It targets all of the files
# in the "AirfoilPressureData" folder and generates
# files with the latent data in the
# "AutoencoderLatentSpaceValues" folder.

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def main():

    if os.path.isdir(str(f"{rootPath}{savingExtension}")) == False: # Contingency for latent folder not existing
        os.mkdir(str(f"{rootPath}{savingExtension}")) # Creates the latent data folder
    
    # Add code that wipes out the data in the folder if it already exists?
    # Might be worthwhile in the event of file names changing...

    targetFiles = os.listdir(rootPath + labelExtension) # Lists all Pressure Data files

    encoder = tf.keras.models.load_model(rootPath + encoderExt) # Loads the Encoder model made earlier

    for ID, file in enumerate(targetFiles): # Iterates through all the target files

        print("Saving File #" + str(ID) + ", Path: " + file) # Debug printout

        fileData = np.loadtxt(rootPath + labelExtension + "/" + file) # Adds file path data to the filename

        fileData = np.reshape(fileData, [1, imageShape[0], imageShape[1]]) # This adds a empty dimension to the start of the pressure data,
        # converting a 84 x 292 to a 1 x 84 x 292. This is important as the prediction function expects a known batch size, in this case 1.

        encodedData = encoder.predict(fileData) # Generates the latent space data

        filePath = rootPath + savingExtension + "/" + file # Saving file path, not the same as the pressure data path

        np.savetxt(filePath, encodedData, fmt = '%s') # Saves the data to the latent folder


if __name__ == "__main__":
        
    rootPath = os.path.dirname(__file__) # Defines the root path.
    rootPath = rootPath + '/' 

    savingExtension = 'AutoencoderLatentSpaceValues'
    labelExtension = 'AirfoilPressureData' # Important file paths. Feel free to change.
    encoderExt = 'encoder.keras'

    imageShape = [84, 292] # Shape of the pressure data. Turbulent dataset is 84 x 292. 

    main()