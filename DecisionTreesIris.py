

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics



"""
Data Engineering and Analysis
"""
#Load the dataset

iris_data = pd.read_csv("iris.csv")

iris_data.dtypes
iris_data.describe()
iris_data.head()

"""
1. The ranges of values in each of the variables (columns) look ok without any kind of outliers

2. There is equal distribution of the three classes - setosa, versicolor and virginia

No cleansing required

"""
#Exploratory Data Analysis

plt.scatter(iris_data['Petal.Length'],iris_data['Petal.Width'])
plt.cla()
plt.scatter(iris_data['Sepal.Length'], iris_data['Sepal.Width'])
plt.cla()

plt.boxplot([[iris_data['Petal.Length'][iris_data.Species=='setosa']],
              [iris_data['Petal.Length'][iris_data.Species=='versicolor']] ,
                [iris_data['Petal.Length'][iris_data.Species=='virginica']] ],
            labels=('setosa','versicolor','virginica'))
plt.cla()            

plt.boxplot([[iris_data['Petal.Width'][iris_data.Species=='setosa']],
              [iris_data['Petal.Width'][iris_data.Species=='versicolor']] ,
                [iris_data['Petal.Width'][iris_data.Species=='virginica']] ],
            labels=('setosa','versicolor','virginica'))
            
plt.cla()
plt.boxplot([[iris_data['Sepal.Length'][iris_data.Species=='setosa']],
              [iris_data['Sepal.Length'][iris_data.Species=='versicolor']] ,
                [iris_data['Sepal.Length'][iris_data.Species=='virginica']] ],
            labels=('setosa','versicolor','virginica'))
            
plt.cla()
plt.boxplot([[iris_data['Sepal.Width'][iris_data.Species=='setosa']],
              [iris_data['Sepal.Width'][iris_data.Species=='versicolor']] ,
                [iris_data['Sepal.Width'][iris_data.Species=='virginica']] ],
            labels=('setosa','versicolor','virginica'))
            
"""
All 3 except Sepal Width seem to bring the significant differenciation
 between the 3 classes
"""
#Correlations
iris_data.corr()

"""
Modeling and Prediction
"""
#Split into training and testing sets

predictors = iris_data[['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width']]
targets = iris_data.Species

pred_train, pred_test, tar_train, tar_test  =   train_test_split(predictors, targets, test_size=.3)

pred_train.shape
pred_test.shape
tar_train.shape
tar_test.shape

#Build model on training data
classifier=DecisionTreeClassifier()
classifier=classifier.fit(pred_train,tar_train)

predictions=classifier.predict(pred_test)

sklearn.metrics.confusion_matrix(tar_test,predictions)
sklearn.metrics.accuracy_score(tar_test, predictions)
sklearn.metrics.classification_report(tar_test, predictions)

#Displaying the decision tree
from sklearn import tree
from StringIO import StringIO
from IPython.display import Image
out = StringIO()
tree.export_graphviz(classifier, out_file=out)
import pydot
graph=pydot.graph_from_dot_data(out.getvalue())
Image(graph.create_png())

"""
The model shows very high accuracy. The reason why the accuracy
 is so high is because, the data itself has very strong signals 
 (seperation between the classes). Sepal.Length and Sepal.Width
 have very high correlations and they are used in the decision
 tree. In order to see how the tree will behave if it only had
 Sepal.Length and Sepal.Width, let us remove that data and see
 how accurate the tree is.
"""
#Split into training and testing sets

#Only pick 2 features
predictors = iris_data[['Sepal.Length','Sepal.Width']]
targets = iris_data.Species

pred_train, pred_test, tar_train, tar_test  =   train_test_split(predictors, targets, test_size=.3)

pred_train.shape
pred_test.shape
tar_train.shape
tar_test.shape

#Build model on training data
classifier=DecisionTreeClassifier()
classifier=classifier.fit(pred_train,tar_train)

predictions=classifier.predict(pred_test)

sklearn.metrics.confusion_matrix(tar_test,predictions)
sklearn.metrics.accuracy_score(tar_test, predictions)
sklearn.metrics.classification_report(tar_test, predictions)


"""
There is a big drop in accuracy score to 60% from 90% when
top predictor variables are removed from the dataset.
"""