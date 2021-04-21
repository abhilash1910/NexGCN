# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 00:18:44 2020

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






#G=nx.MultiGraph()
#G.add_weighted_edges_from([(1,2,.5), (1,2,.75), (2,3,.5)])
G=nx.DiGraph()
G.add_weighted_edges_from([(0,2,0.5), (3,0,0.75),(3,2,0.75)])
nx.draw(G)
plt.show()
exp=venom.ExperimentalGCN()
kernel=venom.feature_kernels()
X=kernel.centrality_kernel(katz_centrality,G)
#X=kernel.feature_random_weight_kernel(24,G)
#X=kernel.feature_distributions(np.random.poisson(6,10),G)
#print(X)
exp.create_network(G,X,True)

predictions=exp.extract_features(128,2,keras.activations.sigmoid,'adam',5,20,1)
print(predictions)
exp.draw_graph(predictions,exp.network.F.shape[-1],300,False,90,90)
output_class=exp.get_outcome(2)

print(output_class)
