# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 09:25:42 2020

@author: cnyy
"""

from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

#导入结果向量
df=pd.read_csv('A.csv',header=None)

#归一化
X=preprocessing.minmax_scale(df)

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

#%%
# Applying KMeans to the dataset with the optimal number of cluster
kmeans=KMeans(n_clusters= 6, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
Y_Kmeans = kmeans.fit_predict(X)

# Visualising the clusters
for i in range(0,6):
    plt.scatter(X[Y_Kmeans == i, 0], X[Y_Kmeans == i,1],s = 5,  alpha=0.2, label = 'Cluster'+str(i+1))

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 50, c = 'yellow', label = 'Centroids')

plt.title('Clusters of pipelines')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

#%%
#取出各簇中的管段编号
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

#%%
#导入传统风险值，并按编号顺序重排
d=pd.read_csv('风险值.csv',header=None)
dd=d.sort_values(by=0, ascending=True)
c1=[]
c2=[]
c3=[]
c4=[]
c5=[]
c6=[]

#%% 
#计算每个簇内平均风险值
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

#4>1>6>2>3>5
#c=np.concatenate((c1,c2,c3,c4,c5,c6),axis=0)

#%%利用networkx库绘制管道风险等级颜色分级图；数据点上千时效果较差，可用arcgis可视化

#打开之间保存的图邻接列表和节点位置文件，重现图
ba=nx.read_adjlist("1adj.txt",nodetype=int)
pos2= np.load('1pos.npy').item()
#nx.draw_networkx(ba,pos2,with_labels = False, node_size = 10)

#传统模型连续着色
A=nx.draw_networkx_nodes(ba, pos2, a0, node_size = 1,node_color = c1,cmap='Reds')
nx.draw_networkx_nodes(ba, pos2, a1, node_size = 1,node_color = c2,cmap='Reds')
nx.draw_networkx_nodes(ba, pos2, a2, node_size = 1,node_color = c3,cmap='Reds')
nx.draw_networkx_nodes(ba, pos2, a3, node_size = 1,node_color = c4,cmap='Reds')
nx.draw_networkx_nodes(ba, pos2, a4, node_size = 1,node_color = c5,cmap='Reds')
nx.draw_networkx_nodes(ba, pos2, a5, node_size = 1,node_color = c6,cmap='Reds')

plt.title("Risk Level Of Pipeline Network")
plt.colorbar(A)

nx.draw_networkx_edges(ba, pos2, alpha=0.2)


#机器学习模型离散着色
'''
nx.draw_networkx_nodes(ba, pos2, a3, node_size = 1,node_color = '#4a0100',label='RL6',alpha=0.3)
nx.draw_networkx_nodes(ba, pos2, a2, node_size = 1,node_color = '#9a0200',label='RL5',alpha=0.3)
nx.draw_networkx_nodes(ba, pos2, a0, node_size = 1,node_color = '#e50000',label='RL4',alpha=0.3)
nx.draw_networkx_nodes(ba, pos2, a5, node_size = 1,node_color = '#fd5956',label='RL3',alpha=0.3)
nx.draw_networkx_nodes(ba, pos2, a1, node_size = 1,node_color = '#ff9a8a',label='RL2',alpha=0.3)
nx.draw_networkx_nodes(ba, pos2, a4, node_size = 1,node_color = '#ffb19a',label='RL1',alpha=0.3)

nx.draw_networkx_edges(ba, pos2, alpha=0.2)

plt.title("Risk Level Of Pipeline Network")
plt.legend()
plt.show()
'''
#%% 输出管段簇和编号
np.savetxt('1.txt',a0,fmt="%d")
np.savetxt('2.txt',a1,fmt="%d")
np.savetxt('3.txt',a2,fmt="%d")
np.savetxt('4.txt',a3,fmt="%d")
np.savetxt('5.txt',a4,fmt="%d")
np.savetxt('6.txt',a5,fmt="%d")







    
