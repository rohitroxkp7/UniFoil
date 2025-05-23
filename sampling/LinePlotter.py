# LinePlotter

# Plots the predicted and real values on a y = x line

import os
import pyvista as pv
import niceplots
import matplotlib.pyplot as plt

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams["font.family"] = "Times New Roman"
plt.style.use(niceplots.get_style())

from matplotlib import colormaps
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def normalize(inputArray): # Convert all values to fall between 0 and 1. Good for training.

    normalizedArray = 1 * ((inputArray - np.min(inputArray)) / (np.max(inputArray) - np.min(inputArray)))

    return normalizedArray, np.min(inputArray), np.max(inputArray)

def denormalize(inputArray, minValue, maxValue): # Convert values back to real space. Important for algorithms running on the real space data.

    denormalizedArray = (inputArray * (maxValue - minValue)) + minValue

    return denormalizedArray

def l2Norm(real, pred):
    norm = np.linalg.norm(pred - real) / np.linalg.norm(real)
    return norm


def getPressureDataCGNS(filePath):
    reader = pv.CGNSReader(filePath)
    reader.load_boundary_patch = False
    ds = reader.read()

    # Extract all blocks
    all_blocks = []
    def extract_blocks(mb):
        for i in range(mb.n_blocks):
            block = mb[i]
            if isinstance(block, pv.MultiBlock):
                extract_blocks(block)
            elif block is not None:
                all_blocks.append(block)
    extract_blocks(ds)

    if(len(all_blocks) < 4): # In the event of a CGNS having a valid name but invalid amount of blocks, abort the program
        return

    # Access Block 3
    block3 = all_blocks[3]

    # Extract block data
    pressureCoefficent = block3.cell_data['CoefPressure']

    # Reshape the first N rings
    n_rings = imageShape[0]
    n_pts_per_ring = imageShape[1]
    total = n_rings * n_pts_per_ring

    #vel_mag_reshaped = vel_mag[:total].reshape((n_rings, n_pts_per_ring))
    pressureCoefReshaped = pressureCoefficent[:total].reshape((n_rings, n_pts_per_ring))

    return pressureCoefReshaped

def main(examples, isRandom):

    print("Loading Models...")

    decoder = tf.keras.models.load_model(rootPath + decoderExt)

    print("Decoder Loaded.")

    dnn = tf.keras.models.load_model(rootPath + dnnExt)

    print("DNN Loaded.")

    predictionLabels = []

    groundTruthLabels = []

    targetFiles = os.listdir(rootPath + featureExtension + '/')

    if isRandom == True:
        targetFiles = random.sample(targetFiles, k = examples)

    print(f"Generating {examples} example airfoils.")

    for file in targetFiles[:examples]: # Note for later: Implement De-Normalizing!

        print(file)

        filePath = rootPath + featureExtension + '/' + file

        prediction = np.loadtxt(filePath)

        prediction = prediction[:-2]

        latentSpaceValues = dnn.predict(np.reshape(prediction, [1, len(prediction)]))

        latentSpaceValues = np.reshape(latentSpaceValues, [latentDim,])

        minMax = latentSpaceValues[-2:]

        latentSpaceValues = latentSpaceValues[:-2] # Remove the last two values from the latent space.

        outputValues = decoder.predict(np.reshape(latentSpaceValues, [1, len(latentSpaceValues)]))

        outputValues = denormalize(outputValues, -1 * minMax[0], minMax[1])

        outputValues = np.reshape(outputValues, [1,-1])

        predictionLabels.append(outputValues)

        targetCGNS = rootPath + generationExtension + '/' + file[:-4] + cgnsExt

        groundTruth = getPressureDataCGNS(targetCGNS)
        
        groundTruthLabels.append(np.reshape(groundTruth, [1, -1]))


    predictionLabels = np.vstack(predictionLabels)

    groundTruthLabels = np.vstack(groundTruthLabels)

    print(np.shape(predictionLabels))
    print(np.shape(groundTruthLabels))

    fig, axes = plt.subplots(1,1)

    axes.scatter(groundTruthLabels[0,:], predictionLabels[0,:], s = 2.5, c = 'orange')

    axes.axline([-0.5, -0.5], [0.5, 0.5], c = 'blue')

    axes.legend(["$C_p$", "y = x"], loc = 2, fontsize = 20)

    axes.set_xlabel("Ground Truth Values", size = 20)

    axes.set_ylabel("Prediction Values", size = 20, rotation = 0, labelpad = 75)

    plt.axis('square')

    axes.tick_params(axis='both', which='major', labelsize = 18)

    fig.set_size_inches(10, 8)

    #plt.show()

    plt.savefig(rootPath + "truthVsPrediction.pdf", bbox_inches ='tight')


if __name__ == "__main__":

    rootPath = os.path.dirname(__file__)
    rootPath = rootPath + '/'

    featureExtension = "AirfoilFeatureData"
    generationExtension = "output"
    decoderExt = "decoder.keras"
    dnnExt = "dnn.keras"
    cgnsExt = "_turb.cgns"

    imageShape = [84, 292]

    latentDim = int(imageShape[0] * imageShape[1] * 0.25) + 2

    exampleCount = 6
    randomStatus = True

    main(exampleCount, randomStatus)