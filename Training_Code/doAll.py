# Airfoil Training System
# Ben Melanson
# May 20th, 2025

# Description
# This code should run all of the required operations back to back. It should be robust enough
# to create multiple seperate folders for each required step. 

# Current Steps:
# 1.) Reshape Data
# 2.) Train Encoder Model
# 3.) Encode all of the data
# 4.) Append Min/Max
# 5.) Train DNN Model
# 6.) Decode Example(s)


# Importing all the subprocesses

import os
import Reshaper
import ConvolutionalNetwork
import Encoder
import Appender
import Dnn
import Decoder


# Important Definitions

geometryFeatures = 14 # Geometry Modes, 14 for the Turbulent dataset and 30 for the Laminar Dataset
additionalFeatures = 3 # Mach Number, Angle of Attack, and Reynolds Number. Dont change this value.

featureCount = geometryFeatures + additionalFeatures

imageShape = [84, 292]

generationExtension = 'output' # The folder that contains all the .cgns files.
featureExtension = 'AirfoilFeatureData' # The folder that will contain the feature data.
labelExtension = 'AirfoilPressureData' # The folder that will contain the pressure data.
savingExtension = 'AutoencoderLatentSpaceValues' # The folder that contains the latent space data.

encoderExt = 'encoder.keras'
decoderExt = 'deceoder.keras'
dnnExt = 'dnn.keras'
cgnsExt = "_turb.cgns"

latentDim = int(imageShape[0] * imageShape[1] * 0.25) + 2 # Make sure that this is actually an integer

# 1.) Run Data Reshaper
# The data reshaper converts the agreed upon .CGNS file standard into more convienient file paths.
# It also extracts the required data into specified directories.
# Note: The data reshaper ONLY HAS TO RUN ONCE!

# Check if target directories exist

rootPath = os.path.dirname(__file__)
rootPath = rootPath + '/'

forceReshape = False

if forceReshape == False:
    if os.path.isdir(rootPath + featureExtension) == True & os.path.isdir(rootPath + labelExtension) == True:
        print("Reshaped Data Detected, skipping reshaping...")

    else:
        os.mkdir(rootPath + featureExtension)
        os.mkdir(rootPath + labelExtension)

        print("Preparing Reshaper.")

        Reshaper.main()

        print("Reshaping Complete.")
else:
    os.mkdir(rootPath + featureExtension)
    os.mkdir(rootPath + labelExtension)

    print("Preparing Reshaper.")

    Reshaper.main()

    print("Reshaping Complete.")


# 2.) Train the Encoder
# The encoder is a 

print("Preparing Encoder.")

epochCount = 100

ConvolutionalNetwork.main(epochCount)

print("Convolutional Network Training Complete.")


# 3.) Encode Pressure Data
# The data appender copies the minimum and maximum values used in normalization to the latent space data.

print("Encoding Pressure Data.")

Encoder.main()

print("Encoding Complete.")

# 4.) Run Data Appender
# The data appender copies the minimum and maximum values used in normalization to the latent space data.
# I am not certain if this is still needed but I am going to keep it around just incase.

print("Appending Values.")

Appender.main()

print("Appending Complete.")


# 4.) Train DNN
# Trains the Deep Neural Network using the provided Epoch Count. Make sure to configure the DNN to use the
# Correct 

print("Preparing DNN")

epochCount = 100

Dnn.main(epochCount)

print("DNN Training Complete")


# 5.) Decode examples
# Set the Example Count to determine how many random examples you want to generate.
# Generates examples from the path defined in Decoder.py.
# Do NOT change the path settings in Decoder.py if you want it to keep working.

print("Printing Examples")

exampleCount = 4

randomStatus = True

Decoder.main(exampleCount, randomStatus)

print("All operations completed! :)")
