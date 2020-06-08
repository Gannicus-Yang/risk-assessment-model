# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 09:25:42 2020

@author: cnyy
"""
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#导入数据
df1=pd.read_csv('1.csv')

#标准化
#df_1=preprocessing.scale(df)

#归一化
x=preprocessing.minmax_scale(df1)

#%% 降维
pca=PCA(n_components=2)
X=pca.fit_transform(x)

#%% 层次聚类
from sklearn.cluster import AgglomerativeClustering
Y_agg = AgglomerativeClustering(n_clusters=6).fit_predict(X)
plt.scatter(X[:,0], X[:,1], s=5, c=Y_agg, alpha=0.5)
plt.show()

#%% DBSCAN
from sklearn.cluster import DBSCAN
Y_DBSCAN=DBSCAN(eps=0.2,min_samples=10).fit_predict(X)
#eps=0.1,min_samples=10
plt.scatter(X[:,0], X[:,1],s=5,c=Y_DBSCAN, alpha=0.5)
plt.show()
#对凸数据集效果很差

#%% K-Means 肘部法则
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
kmeans=KMeans(n_clusters= 6, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
Y_Kmeans = kmeans.fit_predict(X)

# Visualising the clusters
for i in range(0,6):
    plt.scatter(X[Y_Kmeans == i, 0], X[Y_Kmeans == i,1],s = 5,  alpha=0.2, label = 'Cluster'+str(i+1))

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 50, c = 'yellow', label = 'Centroids')

plt.title('Clusters of pipelines')
plt.xlabel('pca x')
plt.ylabel('pca y')
plt.legend()
plt.show()

#%% 取出各簇中的管段编号
a0=[]
a1=[]
a2=[]
a3=[]
a4=[]
a5=[]

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

#%% 导入传统风险值，并按编号顺序重排
d=pd.read_csv('风险值.csv',header=None)
dd=d.sort_values(by=0, ascending=True)
c1=[]
c2=[]
c3=[]
c4=[]
c5=[]
c6=[]

#%% 计算每个簇内平均风险值
for i in range(len(a0)):
    c1.append(d[1][a0[i]])
c1=np.array(c1)
rc1=c1.mean()

for i in range(len(a1)):
    c2.append(d[1][a1[i]])
c2=np.array(c2)
rc2=c2.mean()

for i in range(len(a2)):
    c3.append(d[1][a2[i]])
c3=np.array(c3)
rc3=c3.mean()

for i in range(len(a3)):
    c4.append(d[1][a3[i]])
c4=np.array(c4)
rc4=c4.mean()

for i in range(len(a4)):
    c5.append(d[1][a4[i]])
c5=np.array(c5)
rc5=c5.mean()

for i in range(len(a5)):
    c6.append(d[1][a5[i]])
c6=np.array(c6)
rc6=c6.mean()

#4>=1>6>2>3>5
#4>1>6>2>3>5
#c=np.concatenate((c1,c2,c3,c4,c5,c6),axis=0)

#%% 输出各簇的管道编号
np.savetxt('1.txt',a0,fmt="%d")
np.savetxt('2.txt',a1,fmt="%d")
np.savetxt('3.txt',a2,fmt="%d")
np.savetxt('4.txt',a3,fmt="%d")
np.savetxt('5.txt',a4,fmt="%d")
np.savetxt('6.txt',a5,fmt="%d")







    
