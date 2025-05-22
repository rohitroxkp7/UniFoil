# Data Reshaper
# Ben Melanson
# May 17th, 2025

# Description
# This script uses the PyVista library to extract values
# from a .cgns file. This is a lightweight alternative to
# other data handling methods.

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import os

rootPath = os.path.dirname(__file__)
rootPath = rootPath + '/'

airfoilFeatureData = np.loadtxt(str(f"{rootPath}airfoilData.dat"))

airfoilPropertieData = np.loadtxt(str(f"{rootPath}Airfoil_Case_Data_turb.csv"), skiprows = 1, delimiter = ",")

def normalize(inputArray): # Convert all values to fall between 0 and 1. Good for training.

    normalizedArray = 1 * ((inputArray - np.min(inputArray)) / (np.max(inputArray) - np.min(inputArray)))

    return normalizedArray, np.min(inputArray), np.max(inputArray)

def saveFileData(file, index): # This function extracts the pressure data from a .cgns given the file name and index.

    # Generates and saves the target file data.
    # File is the raw file name for the .cgns.
    # Index is the ID of the .cgns, required for pulling property data from a table.

    checkforErrorString = file.rsplit('_')[0]

    airfoilSavingName = file[:-10]

    if checkforErrorString != "airfoil": # Sometime .cgns files are not generated correctly, abort the program if they dont follow the standard naming convention.
        return

    if os.path.isfile(str(f"{rootPath}{featureExtension}/{airfoilSavingName}.csv")) and os.path.isfile(str(f"{rootPath}{labelExtension}/{airfoilSavingName}.csv")):
        print("Data already exists, skipping...")
        return

    splitString = file.rsplit('_')[1]
    
    filepath = str(f"{rootPath}{generationExtension}/{file}")

    reader = pv.CGNSReader(filepath) # Creates a PyVista CGNS Reader pointed at the given file.
    
    reader.load_boundary_patch = False

    cgnsData = reader.read() # Reads the data from the .cgns.

    all_blocks = [] # Empty array for storing the block data.

    def extract_blocks(blockReader): # This function extracts the data blocks from the .cgns (Should this function be nested in here?)
        for i in range(blockReader.n_blocks):
            block = blockReader[i]
            if isinstance(block, pv.MultiBlock):
                extract_blocks(block)
            elif block is not None:
                all_blocks.append(block)
    
    extract_blocks(cgnsData)

    if(len(all_blocks) < 4): # In the event of a CGNS having a valid name but invalid amount of blocks, abort the program.
        print("Error #1: Improperly Formatted CGNS!")
        return

    block3 = all_blocks[3] # Access Block 3 which stores the data for some reason, not quite sure why.

    pressureCoefficent = block3.cell_data['CoefPressure'] # Extracts the pressure data from the block.

    n_rings = imageShape[0]
    n_pts_per_ring = imageShape[1]
    total = n_rings * n_pts_per_ring

    pressureCoefReshaped = pressureCoefficent[:total].reshape((n_rings, n_pts_per_ring)) # Reshapes the pressure data into the image shape.

    normalizedPressureCoef, minPressureCoef, maxPressureCoef = normalize(pressureCoefReshaped)

    if np.isnan(normalizedPressureCoef).any(): # In the event of a CGNS Pressure Data having NAN values, abort the program.
        print("Error #2: CGNS Contains NAN values!")
        return
    
    invertedMinPressureCoef = -1.0 * minPressureCoef

    airfoil = int(splitString) - 1 # Airfoils are numbered starting at 1 but arrays start at 0. Offsets by 1 to line up these values.

    fileID = index

    fileMachNumber = airfoilPropertieData[fileID, 2] # Pulls the Mach Number from the properties file.

    fileAOA = airfoilPropertieData[fileID, 3] # Pulls the Angle of Attack from the properties file.

    fileReynoldsNumber = (airfoilPropertieData[fileID, 4]) # Pulls the Reynolds Number from the properties file.

    fileReynoldsNumber = fileReynoldsNumber / (10000000) # Reduces the Reynolds Number to be a float less than 1 (not tested for all edge cases)

    fileFeatureData = airfoilFeatureData[airfoil, :geometryFeatures] # Copy the geometry feature data into a new variable. 

    fileFeatureData = np.append(fileFeatureData, fileMachNumber) # These 5 lines append other critical feature data into the previously defined variable
    fileFeatureData = np.append(fileFeatureData, fileAOA)
    fileFeatureData = np.append(fileFeatureData, fileReynoldsNumber)
    fileFeatureData = np.append(fileFeatureData, invertedMinPressureCoef)
    fileFeatureData = np.append(fileFeatureData, maxPressureCoef)

    if len(fileFeatureData) != geometryFeatures + additionalFeatures + 2:
        print("Error #3: Feature File Length Mismatch!") # Contingency for when the feature count doesn't match as expected.
        return
    
    if np.shape(normalizedPressureCoef)[0] != imageShape[0] or np.shape(normalizedPressureCoef)[1] != imageShape[1]: # This feels very inefficent
        print("Error #4: Array Length Mismatch!") # Break case 4: Aborts saving if the reshaped pressure data format is incorrect.
        return # I am not sure if this error case can ever actually trigger due to how I am running the reshaping, but you never know. :|

    np.savetxt(str(f"{rootPath}{featureExtension}/{airfoilSavingName}.csv"), fileFeatureData, fmt='%s')
    np.savetxt(str(f"{rootPath}{labelExtension}/{airfoilSavingName}.csv"), normalizedPressureCoef, fmt='%s')

def main():

    for airfoilID, airfoilProperties in enumerate(airfoilPropertieData):
        
        airfoilSet = airfoilID // 14
        airfoilCase = airfoilID % 14

        airfoilFile = str(f"airfoil_{str(airfoilSet + 1)}_G2_A_L0_case_{airfoilCase}_000_surf{cgnsExt}") # This might need to be changed depending on the file structure

        airfoilFilePath = str(f"{rootPath}{generationExtension}/{airfoilFile}")

        if os.path.isfile(airfoilFilePath):
            print(f"Saving Simulation #{airfoilID + 1}, located in Airfoil #{airfoilSet + 1}")
            saveFileData(airfoilFile, airfoilID)
        else:
            print(f"Simulation #{airfoilID + 1} not found!")


if __name__ == '__main__':

    rootPath = os.path.dirname(__file__)
    rootPath = rootPath + '/'
    
    imageShape = [84, 292]

    geometryFeatures = 14

    additionalFeatures = 3

    generationExtension = 'output'
    featureExtension = 'AirfoilFeatureData'
    labelExtension = 'AirfoilPressureData'

    cgnsExt = "_turb.cgns"

    main()