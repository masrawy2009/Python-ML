
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt

"""
Data Engineering and Analysis
"""
#Load the dataset
auto_data = pd.read_csv("auto-data.csv")
auto_data.dtypes
auto_data.describe()
auto_data.head()

#Look at scatter plots
plt.scatter(auto_data.HP, auto_data.PRICE)
plt.cla() 
plt.scatter(auto_data['MPG-CITY'], auto_data['MPG-HWY'])
plt.cla()

#Center and scale
from sklearn import preprocessing
auto_data['HP']=preprocessing.scale(auto_data['HP'].astype('float64'))
auto_data['RPM']=preprocessing.scale(auto_data['RPM'].astype('float64'))
auto_data['MPG-CITY']=preprocessing.scale(auto_data['MPG-CITY'].astype('float64'))
auto_data['MPG-HWY']=preprocessing.scale(auto_data['MPG-HWY'].astype('float64'))
auto_data['PRICE']=preprocessing.scale(auto_data['PRICE'].astype('float64'))
auto_data.describe()


from sklearn.cluster import KMeans

model=KMeans(n_clusters=4)
model.fit(auto_data[['HP','PRICE']][0:100])
prediction=model.predict(auto_data[['HP','PRICE']][0:100])
prediction

plt.cla()
plt.scatter(auto_data['HP'][0:100],auto_data['PRICE'][0:100], c=prediction)

"""
Clustering for 5 numeric columns and all rows
"""
model=KMeans(n_clusters=4)
model.fit(auto_data.loc[0:,'HP':'PRICE'])
prediction=model.predict(auto_data.loc[0:,'HP':'PRICE'])
prediction

"""
Finding optimal number of clusters by repeating clustering
for various values of k
"""
from scipy.spatial.distance import cdist
clusters=range(1,10)
meanDistortions=[]

for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(auto_data.loc[0:,'HP':'RPM'])
    prediction=model.predict(auto_data.loc[0:,'HP':'RPM'])
    meanDistortions.append(sum(np.min(cdist(auto_data.loc[0:,'HP':'RPM'], model.cluster_centers_, 'euclidean'), axis=1)) / auto_data.loc[0:,'HP':'RPM'].shape[0])

plt.cla()
plt.plot(clusters, meanDistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average distortion')
plt.title('Selecting k with the Elbow Method')
