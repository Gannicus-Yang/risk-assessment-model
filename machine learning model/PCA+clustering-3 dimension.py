# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 16:05:41 2020

@author: cnyy
"""
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt

#取出数据；预处理
df=pd.read_csv('1.csv')
df_1=preprocessing.minmax_scale(df)

#降维
pca=PCA(n_components=3)
X=pca.fit_transform(df_1)

#%% K-Means
# Using the elbow method to find the optimal number of clusters
wcss =[]
for i in range (1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter =300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the graph to visualize the Elbow Method to find the optimal number of cluster  
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#%% K-Means 可视化
# Applying KMeans to the dataset with the optimal number of cluster
kmeans=KMeans(n_clusters= 7, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
Y_Kmeans = kmeans.fit_predict(X)

from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
# Visualising the clusters
fig = plt.figure()
ax = Axes3D(fig)

for i in range(0,7):
    ax.scatter(X[Y_Kmeans == i, 0], X[Y_Kmeans == i,1],X[Y_Kmeans == i,2],s = 5,  label = 'Cluster'+str(i+1))
             
ax.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],kmeans.cluster_centers_[:,2], s = 50, c = 'yellow', label = 'Centroids')

ax.set_title('Clusters of pipelines')
ax.set_xlabel('pca x')
ax.set_ylabel('pca y')
ax.set_zlabel('pca z')
plt.legend()
plt.show()

#%% 取出各簇中的管段编号
a0=[]
a1=[]
a2=[]
a3=[]
a4=[]
a5=[]
a6=[]

for i in range(len(Y_Kmeans)):
    if Y_Kmeans[i]==0:
        a0.append(i)
    if Y_Kmeans[i]==1:
        a1.append(i)
    if Y_Kmeans[i]==2:
        a2.append(i)
    if Y_Kmeans[i]==3:
        a3.append(i)        
    if Y_Kmeans[i]==4:
        a4.append(i)
    if Y_Kmeans[i]==5:
        a5.append(i)
    if Y_Kmeans[i]==6:
        a6.append(i)
