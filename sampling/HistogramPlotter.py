# Histogram Plotter

import os
import time
import pyvista as pv
import niceplots
import matplotlib.pyplot as plt

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams["font.family"] = "Times New Roman"
plt.style.use(niceplots.get_style())

from matplotlib import colormaps
import numpy as np
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.models import Model

imageShape = [84, 292]

latentDim = int(imageShape[0] * imageShape[1] * 0.25) + 2

def normalize(inputArray): # Convert all values to fall between 0 and 1. Good for training.

    normalizedArray = 1 * ((inputArray - np.min(inputArray)) / (np.max(inputArray) - np.min(inputArray)))

    return normalizedArray, np.min(inputArray), np.max(inputArray)

def denormalize(inputArray, minValue, maxValue): # Convert values back to real space. Important for algorithms running on the real space data.

    denormalizedArray = (inputArray * (maxValue - minValue)) + minValue

    return denormalizedArray

def l2Norm(real, pred):
    norm = np.linalg.norm(pred - real, ord = 2) / np.linalg.norm(real, ord = 2)
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
    #velocity = block3.cell_data['Velocity']
    pressureCoefficent = block3.cell_data['CoefPressure']
    #vel_mag = np.linalg.norm(velocity, axis=1)

    # Get cell center coordinates
    cell_centers = block3.cell_centers()
    coords = cell_centers.points  # shape: (num_cells, 3)

    # Reshape the first N rings
    n_rings = imageShape[0]
    n_pts_per_ring = imageShape[1]
    total = n_rings * n_pts_per_ring

    # Reshape x, y, velocity
    x = coords[:total, 0].reshape((n_rings, n_pts_per_ring))
    y = coords[:total, 1].reshape((n_rings, n_pts_per_ring))
    #vel_mag_reshaped = vel_mag[:total].reshape((n_rings, n_pts_per_ring))
    pressureCoefReshaped = pressureCoefficent[:total].reshape((n_rings, n_pts_per_ring))

    return pressureCoefReshaped, x, y

def main():

    rootPath = os.path.dirname(__file__)
    rootPath = rootPath + '/'

    savingFileExt = "AirfoilFeatureData/"
    cgnsExt = "output/"
    decoderExt = "decoder.keras"
    dnnExt = "dnn.keras"

    """

    startTime = time.time()

    print("Loading Models...")

    decoder = tf.keras.models.load_model(rootPath + decoderExt)

    print("Decoder Loaded.")

    dnn = tf.keras.models.load_model(rootPath + dnnExt)

    print("DNN Loaded.")

    targetFiles = os.listdir(rootPath + savingFileExt)

    l2Values = []

    for fileID, file in enumerate(targetFiles):

        filePath = rootPath + savingFileExt + file

        prediction = np.loadtxt(filePath)

        prediction = prediction[:-2]

        latentSpaceValues = dnn.predict(np.reshape(prediction, [1, len(prediction)]))

        latentSpaceValues = np.reshape(latentSpaceValues, [latentDim,])

        minMax = latentSpaceValues[-2:]

        latentSpaceValues = latentSpaceValues[:-2] # Remove the last two values from the latent space.

        outputValues = decoder.predict(np.reshape(latentSpaceValues, [1, len(latentSpaceValues)]))

        outputValues = np.reshape(outputValues, [imageShape[0], imageShape[1]])

        outputValues = denormalize(outputValues, -1 * minMax[0], minMax[1])

        targetCGNS = rootPath + cgnsExt + file[:-4] + '_turb.cgns'

        groundTruth = getPressureDataCGNS(targetCGNS)[0]

        groundTruth = np.reshape(groundTruth, [imageShape[0], imageShape[1]])

        l2Value = l2Norm(groundTruth, outputValues)

        l2Values = np.append(l2Values, l2Value)

        print("\n \nFile ID: " + str(fileID) + " L2Value: " + str(l2Value))

    print(l2Values)

    np.savetxt(rootPath + "finalPredictions.csv", l2Values, fmt = '%s')

    """

    l2Values = np.loadtxt(rootPath + "finalPredictions.csv")

    fig, axes = plt.subplots(1,1)

    binValues = np.logspace(-2, 1, 50)

    axes.hist(l2Values, bins = binValues, edgecolor = "black", color = 'orange', orientation = 'vertical')

    #axes.set_title("Turbulent Dataset Relative L2 Error")

    axes.set_ylabel("Simulations predicted", size = 20, rotation = 0, labelpad = 125)

    axes.set_xlabel("Average $L_2$ Norm" r"$\left(\frac{\vert\vert{C_{\text{p,real}} - {C_{\text{p,pred}}}}\vert\vert_{L_2}}{\vert\vert{C_{\text{p,real}}}\vert\vert_{L_2}}\right)$", size = 20)

    axes.set_xscale('log')

    axes.tick_params(axis='both', which='major', labelsize = 18)

    fig.set_size_inches(16, 5)

    plt.show()

    #plt.savefig(rootPath + "turbulentHistogram.pdf", bbox_inches ='tight')


if __name__ == "__main__":

    main()