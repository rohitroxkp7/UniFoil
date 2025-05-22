# Decoder
# Ben Melanson
# May 21st, 2025

# Description
# This script runs the entire neural network package on a set of 
# airfoils and compares the results of the network to the true
# values.

import os
import time
import pyvista as pv
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams["font.family"] = "Times New Roman"

import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from matplotlib.patches import Polygon
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
    norm = np.linalg.norm(real - pred) / np.linalg.norm(real)
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

    if(len(all_blocks) < 4): # In the event of a CGNS having a valid name but invalid amount of blocks, abort the program.
        return

    block3 = all_blocks[3]

    # Extract the block data.
    pressureCoefficent = block3.cell_data['CoefPressure']
    
    cell_centers = block3.cell_centers()
    coords = cell_centers.points  

    # Reshape the first N rings into the imageshape.
    n_rings = imageShape[0]
    n_pts_per_ring = imageShape[1]
    total = n_rings * n_pts_per_ring

    # Reshape x, y, and pressure coefficent.
    x = coords[:total, 0].reshape((n_rings, n_pts_per_ring))
    y = coords[:total, 1].reshape((n_rings, n_pts_per_ring))
    pressureCoefReshaped = pressureCoefficent[:total].reshape((n_rings, n_pts_per_ring))

    return pressureCoefReshaped, x, y

def main(examples, isRandom):

    startTime = time.time()

    print("Loading Models...")

    decoder = tf.keras.models.load_model(rootPath + decoderExt)

    print("Decoder Loaded.")

    dnn = tf.keras.models.load_model(rootPath + dnnExt)

    print("DNN Loaded.")

    predictionLabelsExtended = []

    groundTruthLabelsExtended = []

    xArray = []

    yArray = []

    targetFiles = os.listdir(rootPath + featureExtension + '/')

    if isRandom == True:
        targetFiles = random.sample(targetFiles, k = examples)

    print(f"Generating {examples} example airfoils.")

    for file in targetFiles[:examples]:
        
        # This loop handles all of the data processing for each file.

        print(file)

        filePath = rootPath + featureExtension + '/' + file

        prediction = np.loadtxt(filePath) # Loads the feature data from the filepath.

        prediction = prediction[:-2] # Cuts off the normalization data.

        latentSpaceValues = dnn.predict(np.reshape(prediction, [1, len(prediction)])) # Predicts the latent space values using the saved DNN.

        latentSpaceValues = np.reshape(latentSpaceValues, [latentDim,]) 

        minMax = latentSpaceValues[-2:] # Saves the normalization data generated from the DNN for later.

        latentSpaceValues = latentSpaceValues[:-2] # Remove the last two values from the latent space.

        outputValues = decoder.predict(np.reshape(latentSpaceValues, [1, len(latentSpaceValues)])) # Predicts the pressure field data using the Decoder model.

        outputValues = np.reshape(outputValues, [imageShape[0], imageShape[1]]) # Ensures that the output is a grid in the shape of imageShape.

        outputValues = denormalize(outputValues, -1 * minMax[0], minMax[1]) # Denormalizes the predected pressure field using the generated normalization data.

        #outputValues = denormalize(outputValues, -1, 1) # Use this if you want to ignore the normalization data from the DNN

        targetCGNS = rootPath + generationExtension + '/' + file[:-4] + cgnsExt # Builds the filepath to the actual .cgns using the saved name of the feature data.

        groundTruth, xvals, yvals = getPressureDataCGNS(targetCGNS) # Gets the real pressure field data along with positional coordinates from the .cgns

        groundTruth = np.reshape(groundTruth, [imageShape[0], imageShape[1]]) # Ensures that the ground truth is a grid in the shape of imageShape.

        #groundTruth = normalize(groundTruth)[0] # Use these two lines if you want to ignore the normalization data from the DNN

        #groundTruth = denormalize(groundTruth, -1, 1)

        xvals = np.reshape(xvals, [imageShape[0], imageShape[1]]) # Does the same thing but for the x coordinates.

        yvals = np.reshape(yvals, [imageShape[0], imageShape[1]]) # Does the same thing but for the y coordinates.

        xExtended = np.zeros([imageShape[0], imageShape[1] + 1]) # Creates 4 new empty arrays that are the same as imageShape but with an extra point in the x direction.
        yExtended = np.zeros([imageShape[0], imageShape[1] + 1])
        predExtended = np.zeros([imageShape[0], imageShape[1] + 1])
        truthExtended = np.zeros([imageShape[0], imageShape[1] + 1])

        for i in range(imageShape[0]): # This section appends the first value to the end of each array to prevent a gap from appearing in the plot.
            xExtended[i] = np.append(xvals[i], xvals[i, 0])
            yExtended[i] = np.append(yvals[i], yvals[i, 0])
            predExtended[i] = np.append(outputValues[i], outputValues[i, 0])
            truthExtended[i] = np.append(groundTruth[i], groundTruth[i, 0])

        xArray.append(xExtended) # Saves all the extended datapoints for plotting.
        yArray.append(yExtended)
        predictionLabelsExtended.append(predExtended)
        groundTruthLabelsExtended.append(truthExtended)

    timeElapsed = time.time() - startTime # Determines how long the decoding takes. For me it takes about 5 seconds with 6 airfoils.

    print(f"Processing Complete! Time: {timeElapsed} seconds")

    groundTruthMin = np.min(groundTruthLabelsExtended) # Calculates the min and max values from the ground truth data
    groundTruthMax = np.max(groundTruthLabelsExtended) 
    truthDelta = groundTruthMin + groundTruthMax # Determines which of the min and max has the higher absolute value 

    if truthDelta > 0: # This block here is used to make the colorbars in the plotting section centered at zero.
        groundTruthMin = -1 * groundTruthMax 
    else:
        groundTruthMax = -1 * groundTruthMin

    matplotlib.rc('font', size=6)

    fig, axes = plt.subplots(3, examples + 1, layout = 'constrained')
    for i in range(examples):
        c1 = axes[0, i+1].contourf(xArray[i], yArray[i], groundTruthLabelsExtended[i], vmin = groundTruthMin, vmax = groundTruthMax, levels = np.linspace(groundTruthMin, groundTruthMax, 101), cmap = colormaps["coolwarm"])
        c1.set_edgecolor("face") 
        axes[0, i+1].axis('off')
        axes[0, i+1].set_xlim([-0.5,1.5])
        axes[0, i+1].set_ylim([-1,1])

        localXCoords = xArray[i]
        localYCoords = yArray[i]
        coords = list(zip(localXCoords[0, :imageShape[1]], localYCoords[0, :imageShape[1]]))
        airfoilGeometry = Polygon(coords, facecolor='none', edgecolor='black', linewidth = 1)
        axes[0, i+1].add_patch(airfoilGeometry)

        c2 = axes[1, i+1].contourf(xArray[i], yArray[i], predictionLabelsExtended[i], vmin = groundTruthMin, vmax = groundTruthMax, levels = np.linspace(groundTruthMin, groundTruthMax, 101), cmap = colormaps["coolwarm"])
        c2.set_edgecolor("face")
        axes[1, i+1].axis('off')
        axes[1, i+1].set_xlim([-0.5,1.5])
        axes[1, i+1].set_ylim([-1,1])

        airfoilGeometry = Polygon(coords, facecolor='none', edgecolor='black', linewidth = 1)
        axes[1, i+1].add_patch(airfoilGeometry)

        error = abs((groundTruthLabelsExtended[i] - predictionLabelsExtended[i]))

        l2Error = l2Norm(groundTruthLabelsExtended[i], predictionLabelsExtended[i])

        c3 = axes[2, i+1].contourf(xArray[i], yArray[i], error, vmin = 0.0, vmax = 1.0, levels = np.linspace(0, 1, 101), cmap = colormaps["binary"])
        c3.set_edgecolor("face")
        axes[2, i+1].set_title("{:.3f}".format(l2Error), y = -0.01, fontsize = 12)
        axes[2, i+1].axis('off')
        axes[2, i+1].set_xlim([-0.5,1.5])
        axes[2, i+1].set_ylim([-1,1])

        airfoilGeometry = Polygon(coords, facecolor='none', edgecolor='blue', linewidth = 1)
        axes[2, i+1].add_patch(airfoilGeometry)

    axes[0,0].text(1, 0.5, 'Ground Truth', ha = 'center', va = 'center', size = 18)
    axes[0,0].axis('off')
    axes[1,0].text(1, 0.5, 'Reconstruction', ha = 'center', va = 'center', size = 18)
    axes[1,0].axis('off')
    axes[2,0].text(1, 0.5, "Absolute Error\n"
                    r'$\vert{x_{\text{real}} - x_{\text{pred}}}\vert$', ha = 'center', va = 'center', size = 18)
    axes[2,0].text(1, 0, r'$\text{L}_2$ Error'
                   "\n"
                    r'$\left(\frac{\vert\vert{x_{\text{real}} - {x_{\text{pred}}}}\vert\vert_{L_2}}{\vert\vert{x_{\text{real}}}\vert\vert_{L_2}}\right)$', ha = 'center', va = 'center', size = 15)
    axes[2,0].axis('off')

    cb = plt.colorbar(c1, ax=axes[[0, 1], examples], shrink = 0.5, format=tkr.FormatStrFormatter('%.2f'))
    cb.ax.tick_params(labelsize = 12)
    cb.set_ticks(ticks = [groundTruthMin, groundTruthMin/2, 0, groundTruthMax/2, groundTruthMax])

    cb = plt.colorbar(c3, ax=axes[2, examples], format=tkr.FormatStrFormatter('%.2f'))
    tick_locator = tkr.MaxNLocator(nbins=5)
    cb.locator = tick_locator
    cb.ax.tick_params(labelsize = 12)
    cb.update_ticks()
    
    fig.set_size_inches(12, 6)

    #plt.show()

    plt.savefig(rootPath + "turbulentOutput.pdf")

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