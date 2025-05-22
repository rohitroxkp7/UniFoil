- NN.py is an example to contruct cl/cd surrogates using the training data
- plotairfoil.py is an example to generate the airfoil coordinates using the mode coefficients
- input/training.dat is about 40,000 sample airfoils
- input/validating.dat is about 20,000 sample airfoils
- Each row of training.dat and validating.dat is one sample, and the 67 columns are:
  x(14 modes  mach AoA)  cl    cd     cm     dcl/dx     dcd/dx     dcm/dx
- input/basis.txt is the airfoil modes
- input/bounds.txt is the bounds of the 16 input variables (14 modes + Mach + AoA)

