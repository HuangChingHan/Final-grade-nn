# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 20:48:58 2019

@author: 黃靖涵
"""

# Intialize
from keras.models import Sequential
from keras .layers import Dense
import numpy

# fix random seed for reproducibility
numpy.random.seed(10)

## Step 1 : Load data
# Load sample dataset
dataset = numpy.loadtxt("final_grades.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:16]
Y = dataset[:,16]

## Step 2 : Define Model
# How many layers should we add ? => triial and error
# as first layer in a sequential layer
model = Sequential()
model.add(Dense(12, input_dim=16, activation='relu'))
# the model will take as input arrays of dimention 16
# and output arrays of dim 12

# after the first layer, you don't need to specify 
# the size of the input anymore
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

## Step 3 : Compile Model
# The dataset result will be pass or fail the course, it is a binary classification model.
# So network loss value set it as "binary_crossentropy"
# We only want to see the accuracy rate after evaluating the network,
# so we only put "accuracy" in the metrics. 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


## Step 4 : Fit Model
# Fitting data into the network model.
# iteration (50) with batch_size (10)
model.fit(X, Y, epochs=50, batch_size=10)

## Step 5 : Evaluate Model
# The result will compare the eveluate output with the real output.
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



