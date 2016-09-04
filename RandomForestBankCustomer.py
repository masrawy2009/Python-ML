
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics



bank_data = pd.read_csv("bank.csv", sep=";")
bank_data.dtypes
bank_data.describe()
bank_data.head()

bank_data['age'] = pd.cut(bank_data.age,[1,20,40,60,80,100])
bank_data = bank_data.join(pd.get_dummies(bank_data.marital))
del bank_data['marital']
bank_data.head()

colidx=0
colNames=list(bank_data.columns.values)
for colType in bank_data.dtypes:
    if colType == 'object':
        bank_data[colNames[colidx]]=pd.Categorical.from_array(bank_data[colNames[colidx]]).labels
    colidx= colidx+1
    
bank_data.dtypes
bank_data.describe()
bank_data.corr()


del bank_data['default']
del bank_data['balance']
del bank_data['day']
del bank_data['month']
del bank_data['campaign']
del bank_data['poutcome']


predictors = bank_data[['age','job','education','housing','loan','contact','duration','pdays','previous','divorced','married','single']]
targets = bank_data.y

pred_train, pred_test, tar_train, tar_test  =   train_test_split(predictors, targets, test_size=.3)

pred_train.shape
pred_test.shape
tar_train.shape
tar_test.shape

from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier(n_estimators=25)
classifier=classifier.fit(pred_train,tar_train)

predictions=classifier.predict(pred_test)

sklearn.metrics.confusion_matrix(tar_test,predictions)
sklearn.metrics.accuracy_score(tar_test, predictions)
sklearn.metrics.classification_report(tar_test, predictions)


trees=range(25)
accuracy=np.zeros(25)

for idx in range(len(trees)):
   classifier=RandomForestClassifier(n_estimators=idx + 1)
   classifier=classifier.fit(pred_train,tar_train)
   predictions=classifier.predict(pred_test)
   accuracy[idx]=sklearn.metrics.accuracy_score(tar_test, predictions)
   
plt.cla()
plt.plot(trees, accuracy)

