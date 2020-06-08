# -*- coding: utf-8 -*-
"""
Created on Tue May  5 10:28:14 2020

@author: cnyy
"""

from networkx import to_numpy_matrix
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from sklearn import preprocessing
from sklearn import metrics
from sklearn.cluster import KMeans

#读入网络图，边列表来自arcgis
G=nx.read_edgelist("edgelist.txt",nodetype=int)

#获取GCN需要的各种输入矩阵
order = sorted(list(G.nodes()))
A = to_numpy_matrix(G, nodelist=order)
I = np.eye(G.number_of_nodes())
A_hat = A + I
D_hat = np.array(np.sum(A_hat, axis=0))[0]
D_hat = np.matrix(np.diag(D_hat))

#%%检验图节点数；如有没有边连接的节点，需要添加
E=list(G.nodes())
F = list(map(int, E))
for i in range(6589):
    if i not in F:
        print (i)
        
#%% 保存图的邻接列表和节点位置文件
pos = nx.spring_layout(G)
nx.draw_networkx(G,pos,with_labels=False,node_size=1,alpha=0.1)
nx.write_adjlist(G, "1adj.txt")
np.save('1pos.npy', pos) 

#%% 读入特征矩阵
df=pd.read_csv('feature.csv')
#H=df.values
H=preprocessing.minmax_scale(df)

#%% 构造两层GCN
W_1 = np.random.normal(loc=0, scale=1, size=(9, 4))
W_2 = np.random.normal(loc=0, size=(W_1.shape[1], 2))

#三种激活函数，可搭配
def relu(x):
    return (abs(x) + x) / 2

def sigimoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
 pass # TODO: Compute and return softmax(x)
 x = np.array(x)
 x = np.exp(x)
 x.astype('float32')
 if x.ndim == 1:
  sumcol = sum(x)
  for i in range(x.size):
   x[i] = x[i]/float(sumcol)
 if x.ndim > 1:
  sumcol = x.sum(axis = 0)
  for row in x:
   for i in range(row.size):
    row[i] = row[i]/float(sumcol[i])
 return x

#GCN层函数
def gcn_layer1(A_hat, D_hat, X, W):
    B=scipy.linalg.fractional_matrix_power(D_hat,-0.5)
    return relu(B * A_hat * B * X * W) #relu函数会把负值输出0

def gcn_layer2(A_hat, D_hat, X, W):
    B=scipy.linalg.fractional_matrix_power(D_hat,-0.5)
    return (B * A_hat * B * X * W)

H_1 = gcn_layer1(A_hat, D_hat, H, W_1)
H_2 = gcn_layer2(A_hat, D_hat, H_1, W_2)
output = H_2  #6589*2维的图嵌入向量

#%% 使用轮廓系数比较X的好坏，重复上一步和此步，取历次中聚类效果最好的X
X=output
scores=[]
for k in range(4,10):
    labels=KMeans(n_clusters=k,init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0).fit(X).labels_
    score=metrics.silhouette_score(X, labels)
    scores.append(score)

plt.plot(list(range(4,10)),scores)
plt.xlabel('Number of clusters initialized')
plt.ylabel('silhouette score')
 
#%% 查看聚类效果
kmeans=KMeans(n_clusters= 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
Y_Kmeans = kmeans.fit_predict(X)
for i in range(0,5):
    plt.scatter(X[Y_Kmeans == i, 0].tolist(), X[Y_Kmeans == i,1].tolist(),s = 10,  alpha=0.5, label = 'Cluster'+str(i+1))
plt.legend()
plt.show()

#%% 输出结果向量
np.savetxt('A.csv', output, delimiter = ',')

