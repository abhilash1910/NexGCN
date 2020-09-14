# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 00:35:49 2020

@author: 45063883
"""
import networkx as nx
from networkx import karate_club_graph, to_numpy_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten,Embedding,Dropout
from keras.models import Sequential, Model
from keras import initializers, regularizers,activations,constraints
import keras.backend as k
from tensorflow.keras.layers import Layer,Input
from keras.optimizers import Adam
import numpy as np
from networkx import to_numpy_matrix, degree_centrality, betweenness_centrality, shortest_path_length,in_degree_centrality,out_degree_centrality,eigenvector_centrality,katz_centrality,closeness_centrality
import matplotlib.pyplot as plt
import NexGCN as venom

import pandas as pd
df=pd.read_csv("C:/Users/45063883/3D Objects/Final_Data.csv",encoding='unicode_escape')
df.head()
G=nx.from_pandas_edgelist(df[:50],source='Name',target='Type')
print(type(G))
exp=venom.ExperimentalGCN()
kernel=venom.feature_kernels()
#X=kernel.centrality_kernel(None,G)
#X=kernel.feature_random_weight_kernel(24,G)
X=kernel.feature_distributions(np.random.poisson(6,10),G)
print(X)
exp.create_network(G,X,None)

predictions=exp.extract_binary_features(128,2,keras.activations.sigmoid,'adam',5,20,1)
print(predictions)
exp.draw_graph(predictions,exp.network.F.shape[-1],300,False,90,90,'#00FFFF','#0F00FF')
output_class=exp.get_outcome(2)

print(output_class)
