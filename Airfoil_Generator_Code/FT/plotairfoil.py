import numpy as np
import random

basis = np.loadtxt('input/basis.txt')

# the first row is the x-cor
xslice = basis[0,:].copy()

# the rest are the 14 modes
modes = basis[1:,:].copy()
nmode = modes.shape[0]
npts = modes.shape[1]

bounds = np.loadtxt('input/bounds.txt')

coefs= np.zeros(14)
for i in range(14):
    coefs[i] = random.uniform(bounds[i,0],bounds[i,1])
yslice = np.dot(coefs,modes)

f=open('airfoil.plt','w')
for i in range(npts):
    f.write('%.15f %.15f\n'%(xslice[i],yslice[i]))
f.close()

