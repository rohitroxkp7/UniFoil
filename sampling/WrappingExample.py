# Wrapping Example

# Description
# This code shows off how the data is wrapped and unwrapped.

import os
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

def getVelocityDataCGNS(filePath):
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
    velocity = block3.cell_data['Velocity']
    vel_mag = np.linalg.norm(velocity, axis=1)

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
    velocityReshaped = vel_mag[:total].reshape((n_rings, n_pts_per_ring))

    return velocityReshaped, x, y

def main():

    targetCGNS = rootPath + 'output/airfoil_1_G2_A_L0_case_0_000_surf_turb.cgns'

    velocityData, xvals, yvals = getVelocityDataCGNS(targetCGNS)

    velocityData = np.reshape(velocityData, [imageShape[0], imageShape[1]])

    xvals = np.reshape(xvals, [imageShape[0], imageShape[1]])

    yvals = np.reshape(yvals, [imageShape[0], imageShape[1]])

    xExtended = np.zeros([imageShape[0], imageShape[1] + 1])
    yExtended = np.zeros([imageShape[0], imageShape[1] + 1])
    velocityExtended = np.zeros([imageShape[0], imageShape[1] + 1])

    for i in range(imageShape[0]): # This section appends the first value to the end of each array to prevent a gap from appearing in the plot
        xExtended[i] = np.append(xvals[i], xvals[i, 0])
        yExtended[i] = np.append(yvals[i], yvals[i, 0])
        velocityExtended[i] = np.append(velocityData[i], velocityData[i, 0])

    fig, axes = plt.subplots(1, 1)

    #axes.contourf(xExtended, yExtended, velocityExtended, levels = 1000)
    axes.set_xlim([-0.25, 1.25])
    axes.set_ylim([-1, 1])
    axes.axis('off')

    #coords = list(zip(xvals[0, :292], yvals[0, :292]))
    #polygon = Polygon(coords[:292], facecolor='none', edgecolor='red', linewidth = 4)
    #axes.add_patch(polygon)
    cmap = plt.get_cmap('hsv', 20)

    for i in [0, 45, 48, 50, 51, 52, 53, 54, 55]:
        coords = list(zip(xvals[i, :292], yvals[i, :292]))
        polygon = Polygon(coords[:292], facecolor='none', edgecolor = cmap(i%20))
        axes.add_patch(polygon)

    fig.set_size_inches(6, 6)

    #plt.show()

    fig.savefig(rootPath + "wrappingExample1.png", bbox_inches ='tight', dpi = 1000)

    fig, axes = plt.subplots(1, 1)

    #axes.contourf(np.flip(velocityExtended, 1), levels = 1000)
    
    g = 0
    for i in [0, 45, 48, 50, 51, 52, 53, 54, 55]:
        g = g + 1
        axes.axhline(y = g, linewidth = 2, color = cmap(i%20))
    
    axes.axis('off')
    axes.axhline(y = 0, linewidth = 8, color = 'red')

    fig.set_size_inches(6, 6)

    #plt.show()

    fig.savefig(rootPath + "wrappingExample2.png", bbox_inches ='tight', dpi = 1000)

if __name__ == "__main__":
    
    rootPath = os.path.dirname(__file__)
    rootPath = rootPath + '/'

    imageShape = [84, 292]

    main()