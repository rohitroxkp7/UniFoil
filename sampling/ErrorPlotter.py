import numpy as np
import os
import niceplots

import matplotlib.pyplot as plt
import tensorflow as tf

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams["font.family"] = "Times New Roman"
plt.style.use(niceplots.get_style())

rootPath = os.path.dirname(__file__)
rootPath = rootPath + '/'

encoderTrainingData = np.loadtxt(str(f"{rootPath}encodingModelTrainingHistory.csv"), skiprows = 1, delimiter = ',')
denseTrainingData = np.loadtxt(str(f"{rootPath}denseModelTrainingHistory.csv"), skiprows = 1, delimiter = ',')

print(denseTrainingData)
print(np.shape(denseTrainingData))

encoderEpochs =  encoderTrainingData[:, 0]
encoderL2 = encoderTrainingData[:, 2]
encoderL2Real = encoderTrainingData[:, 5]

denseEpochs =  denseTrainingData[:100, 0]
denseError = denseTrainingData[100:, 1]
denseErrorReal = denseTrainingData[100:, 3]

denseError = 1 - denseError

denseErrorReal = 1 - denseErrorReal

fig, axes = plt.subplots(2, 1, sharex = True)

axes[0].plot(encoderEpochs, encoderL2)
#axes[0].plot(encoderEpochs, encoderL2Real)
axes[0].set_ylabel('Convolutional Network\n$L_2$ Error \n' r"$\left(\frac{\vert\vert{C_{\text{p,real}} - {C_{\text{p,pred}}}}\vert\vert_{L_2}}{\vert\vert{C_{\text{p,real}}}\vert\vert_{L_2}}\right)$", size = 20, rotation = 0, labelpad = 10, ha = 'right')
axes[0].tick_params(axis='both', which='major', labelsize = 18)

axes[1].plot(denseEpochs, denseError)
#axes[1].plot(denseEpochs, denseErrorReal)
axes[1].set_xlabel('Epochs', size = 20)
axes[1].set_ylabel('Dense Network\nError\n' r"$\sqrt{C_{\text{p,pred}} - C_{\text{p,real}}}$", size = 20, rotation = 0, labelpad = 10, ha = 'right')
axes[1].tick_params(axis='both', which='major', labelsize = 18)

fig.set_size_inches(10, 10)

plt.show()

#plt.savefig(rootPath + "turbulentError.pdf", bbox_inches ='tight')