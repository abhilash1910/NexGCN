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
from keras.optimizers import Adam
import numpy as np
from networkx import to_numpy_matrix, degree_centrality, betweenness_centrality, shortest_path_length,in_degree_centrality,out_degree_centrality,eigenvector_centrality,katz_centrality,closeness_centrality
import matplotlib.pyplot as plt
import NexGCN as venom


print("===============Hyperparameters===============")
no_op_features=3
no_epochs=50
no_steps_per_epoch=30
no_pred_steps=3
kernel_size=1024
feature_random_weight=34
distance_param=True
node_size=300
with_labels=True
dim_1=90
dim_2=90
activation=keras.activations.softmax
optimizer='adam'


print("================== First Example: Uniform Features ===============")
Gr = nx.gnm_random_graph(70,140)
exp=venom.ExperimentalGCN()
kernel=venom.feature_kernels()
X=kernel.feature_random_weight_kernel(feature_random_weight,Gr)
exp.create_network(Gr,X,distance_param)
predictions=exp.extract_features(kernel_size,no_op_features,activation,optimizer,no_epochs,no_steps_per_epoch,no_pred_steps)
print("Predictions: ",predictions)
output_classes=exp.draw_graph(predictions,Gr.number_of_nodes(),node_size,with_labels,dim_1,dim_2)
node_num=56
unique_output_class=exp.get_outcome(node_num)
print(unique_output_class)
print("======================== Completed Sample ========================")



print("================== Second Example: Centrality Based Features ===============")

Gr = nx.gnm_random_graph(70,140)
exp=venom.ExperimentalGCN()
kernel=venom.feature_kernels()
X=kernel.centrality_kernel(katz_centrality,Gr)
exp.create_network(Gr,X,True)
predictions=exp.extract_features(kernel_size,no_op_features,activation,optimizer,no_epochs,no_steps_per_epoch,no_pred_steps)
print("Predictions: ",predictions)
output_classes=exp.draw_graph(predictions,Gr.number_of_nodes(),node_size,with_labels,dim_1,dim_2)
node_num=56
unique_output_class=exp.get_outcome(node_num)
print(unique_output_class)
print("======================== Completed Sample ========================")

print("================== Third Example: Distribution Based Features ===============")

Gr = nx.gnm_random_graph(70,140)
exp=venom.ExperimentalGCN()
kernel=venom.feature_kernels()
X=kernel.feature_distributions(np.random.poisson(4,9),Gr)
exp.create_network(Gr,X,True)
predictions=exp.extract_features(kernel_size,no_op_features,activation,optimizer,no_epochs,no_steps_per_epoch,no_pred_steps)
print("Predictions: ",predictions)
output_classes=exp.draw_graph(predictions,Gr.number_of_nodes(),node_size,with_labels,dim_1,dim_2)
node_num=56
unique_output_class=exp.get_outcome(node_num)
print(unique_output_class)
print("======================== Completed Sample ========================")


print("================== Fourth Example: Manual Features ===============")

Gr = nx.gnm_random_graph(70,140)
exp=venom.ExperimentalGCN()
no_nodes=Gr.number_of_nodes()
kernel=venom.feature_kernels()
X=np.matrix([
                [np.random.randn(),np.random.randn(),np.random.randn()]
               for j in range(no_nodes)
                ])
exp.create_network(Gr,X,True)
predictions=exp.extract_features(kernel_size,no_op_features,activation,optimizer,no_epochs,no_steps_per_epoch,no_pred_steps)
print("Predictions: ",predictions)
output_classes=exp.draw_graph(predictions,Gr.number_of_nodes(),node_size,with_labels,dim_1,dim_2)
node_num=56
unique_output_class=exp.get_outcome(node_num)
print(unique_output_class)
print("======================== Completed Sample ========================")
