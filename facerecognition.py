from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader
from pylab import ion, ioff, figure, draw, contourf, clf, show, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from random import normalvariate
import numpy as np
from sklearn import datasets

# ---------------------------------------------------------------------
#
# Olivetti dataset -> 400 images ... 10 images / person 
#	There are ten different images of each of 40 distinct subjects. 
#	For some subjects, the images were taken at different times, varying the lighting, 
#	facial expressions (open / closed eyes, smiling / not smiling) and facial details (glasses / no glasses)
#
#  So we have 40 target / 40 person !!!
#    64bit x 64bit images: number of features=64x64=4096
#

olivettiData = datasets.fetch_olivetti_faces()
dataFeatures = olivettiData.data
dataTargets = olivettiData.target

#plt.matshow(olivettiData.images[11], cmap=cm.Greys_r)
#plt.show()
#print dataTargets[11]
#print dataFeatures.shape

dataSet = ClassificationDataSet(4096, 1 , nb_classes=40)

for i in xrange(len(dataFeatures)):
	dataSet.addSample(np.ravel(dataFeatures[i]), dataTargets[i])
	
testData, trainingData = dataSet.splitWithProportion(0.25)

trainingData._convertToOneOfMany()
testData._convertToOneOfMany()

neuralNetwork = buildNetwork(trainingData.indim, 64, trainingData.outdim, outclass=SoftmaxLayer) 
trainer = BackpropTrainer(neuralNetwork, dataset=trainingData, momentum=0.2, learningrate=0.01, verbose=True, weightdecay=0.02)

trainer.trainEpochs(300)
print 'Error (test dataset): ' , percentError(trainer.testOnClassData(dataset=testData), testData['class'])