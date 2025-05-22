# Appender
# Ben Melanson
# May 20th, 2025

# Description
# The Appender adds the latent space values stored in the Feature Data onto
# the end of the latent space data. This could probably be integrated into
# the encoder script but for debugging purposes it is left seperate.

import os
import numpy as np
import multiprocessing

def updateFunction(file):

    latentFilePath = rootPath + latentExtension + '/'
    featureFilePath = rootPath + featureExtension + '/'

    latentDataPath = latentFilePath + file
    featureDataPath = featureFilePath + file

    latentData = np.loadtxt(latentDataPath)
    featureData = np.loadtxt(featureDataPath)

    newLatentData = np.append(latentData, featureData[-2]) # Add the normalization values that are stored in the feature data to the latent data.
    newLatentData = np.append(newLatentData, featureData[-1]) # The same thing but the second value...

    if len(newLatentData) == latentDim: # Check to make sure that the data is the correct length before saving
        np.savetxt(latentDataPath, newLatentData, fmt = '%s')
        print(f"Added normalization data to {file}")

def main():

    latentFilePath = rootPath + latentExtension + '/'

    targetFiles = os.listdir(latentFilePath) # Loads all the latent space files

    pool = multiprocessing.Pool() # Defines a parallel processing pool to speed up appending

    pool.map(updateFunction, targetFiles) # Runs the update function through the parallel processing pool

    print("All files updated.")

if __name__ == '__main__':

    rootPath = os.path.dirname(__file__)
    rootPath = rootPath + '/'

    latentExtension = 'AutoencoderLatentSpaceValues'
    featureExtension = 'AirfoilFeatureData'

    imageShape = [84, 292]

    latentDim = int(imageShape[0] * imageShape[1] * 0.25) + 2
    
    main()