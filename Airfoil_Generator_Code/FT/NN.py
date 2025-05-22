'''
- Train a NN as a prediction model
'''
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,LeakyReLU,Dropout,Flatten
from tensorflow.keras import activations
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
from tensorflow.keras.models import load_model,save_model

traindata = np.loadtxt('./input/training.dat')
testdata = np.loadtxt('./input/validating.dat')
bounds  = np.loadtxt('./input/bounds.txt')

ntest = testdata.shape[0]

ndim = bounds.shape[0]

def normalizeX(oldx):
    newx = oldx.copy()
    for i in range(ndim):
        newx[i] = 2.0*((oldx[i] - bounds[i,0])/(bounds[i,1]-bounds[i,0]) - 0.5)
    return newx

Alldata = []
labelcl = []
labelcd = []
for i in range(traindata.shape[0]):
    tempdata = normalizeX(traindata[i,:ndim])
    Alldata.append(tempdata)        
    labelcl.append(traindata[i,ndim])
    labelcd.append(traindata[i,ndim+1])

testAlldata = []
testlabelcl = []
testlabelcd = []
for i in range(testdata.shape[0]):
    tempdata = normalizeX(testdata[i,:ndim])
    testAlldata.append(tempdata)        
    testlabelcl.append(testdata[i,ndim])
    testlabelcd.append(testdata[i,ndim+1])

modelcl = Sequential([
  Dense(100,activation=activations.tanh),
  Dense(100,activation=activations.tanh),
  Dense(100,activation=activations.tanh),
  Dense(100,activation=activations.tanh)  
])

modelcd = Sequential([
  Dense(100,activation=activations.tanh),
  Dense(100,activation=activations.tanh),
  Dense(100,activation=activations.tanh),
  Dense(100,activation=activations.tanh)  
])

# Compile the model.
modelcl.compile(
  optimizer='adam',
  loss='mse',
  metrics=['accuracy'],
)

# Compile the model.
modelcd.compile(
  optimizer='adam',
  loss='mse',
  metrics=['accuracy'],
)

#X_train,X_test,y_train,y_test=train_test_split(np.array(Alldata),np.array(labels),test_size=0.0,random_state=1)
X_train = np.array(Alldata)
X_cl    = np.array(labelcl)
X_cd    = np.array(labelcd)

X_test = np.array(testAlldata)
X_testcl    = np.array(testlabelcl)
X_testcd    = np.array(testlabelcd)

# Train the model.
modelcl.fit(
  X_train,
  X_cl,
  validation_data=(X_test, X_testcl),
  epochs=10000,
  batch_size=500,
)

# Train the model.
modelcd.fit(
  X_train,
  X_cd,
  validation_data=(X_test, X_testcd),
  epochs=10000,
  batch_size=500,
)
#model.save('model.h5')
#model = load_model('model.h5')


predictcl = modelcl.predict(np.array(testAlldata))
predictcd = modelcd.predict(np.array(testAlldata))

f = open('error_NN.txt','w')
for i in range(ntest):
    mycl = testlabelcl[i]
    mycd = testlabelcd[i]
    estcl = predictcl[i,0]
    estcd = predictcd[i,0]
    errcl = abs(mycl-estcl)
    errcd = abs(mycd-estcd)
    
    f.write('%.15f %.15f %.15f %.15f\n'%(errcl,errcd,mycl,mycd))
f.close()    
    
errordata = np.loadtxt('error_NN.txt')

pertcd = np.linalg.norm(errordata[:,1])/np.linalg.norm(errordata[:,3])*100.0
pertcl = np.linalg.norm(errordata[:,0])/np.linalg.norm(errordata[:,2])*100.0

f = open('percetage.txt','w')
f.write('%.3f\n%.3f\n'%(pertcl,pertcd))
f.close()


