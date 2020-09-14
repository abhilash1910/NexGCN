# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 01:38:00 2020

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
from keras.optimizers import adam
import numpy as np
from networkx import to_numpy_matrix, degree_centrality, betweenness_centrality, shortest_path_length,in_degree_centrality,out_degree_centrality,eigenvector_centrality,katz_centrality,closeness_centrality
import matplotlib.pyplot as plt
import NexGCN as venom

Gr = nx.gnm_random_graph(70,140)
exp=venom.ExperimentalGCN()
kernel=venom.feature_kernels()
#X=kernel.centrality_kernel(katz_centrality,Gr)
X=kernel.feature_random_weight_kernel(34,Gr)
#X=kernel.feature_distributions(np.random.poisson(4,9),Gr)

exp.create_network(Gr,X,True)
# Xs=np.matrix([
#                 [np.random.randn(),np.random.randn(),np.random.randn()]
#                for j in range(exp.network.A.shape[0])
#                 ])
#
# exp.create_network(None,Xs,True)
#
predictions=exp.extract_binary_features(2048,2,keras.activations.sigmoid,'adam',5,20,1)
print(predictions)
exp.draw_graph(predictions,exp.network.F.shape[-1],300,True,90,90,'#00FFFF','#FF00FF')
output_class=exp.get_outcome(37)

print(output_class)

