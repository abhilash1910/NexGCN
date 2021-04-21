# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 01:36:55 2020

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
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()
class GraphNeuralNetwork(Layer):
    def __init__(self, units, input_dim,activation,kernel_initializer,kernel_regularizer,bias_initializer,bias_regularizer,features):
        self.units=units
        self.input_dim=input_dim
        self.kernel_initializer=initializers.get(kernel_initializer)
        self.bias_initializer=initializers.get(bias_initializer)
        self.kernel_regularizer=regularizers.get(kernel_regularizer)
        self.bias_regularizer=regularizers.get(bias_regularizer)
        self.activation=activations.get(activation)
        self.features=features
        super(GraphNeuralNetwork, self).__init__()
        self.w = self.add_weight(
            shape=(self.input_dim, self.units), initializer=self.kernel_initializer,regularizer=self.kernel_regularizer, trainable=True
        )
        self.b = self.add_weight(shape=(self.units,), initializer=self.bias_initializer,regularizer=self.bias_regularizer, trainable=True)
        
        
    def call(self,inp):
        temp=k.dot(inp,self.features)
        output=k.dot(temp,self.w)
        return self.activation(output + self.b)


class Network():
    #provide networkx graph
    def build(self,G,X,distance):
        if  G is not None:
            self.G=G
        
        if G is None:
            self.G= karate_club_graph()
        self.order=sorted(list(self.G.nodes()))
        self.A=to_numpy_matrix(self.G,nodelist=self.order)
        self.I=np.eye(self.G.number_of_nodes())
        self.X=X
        if X is None: 
            if distance == True:
                'Experimental'
                self.X = np.zeros((self.A.shape[0], 2))
                
                node_distance_instructor = shortest_path_length(self.G, target=self.A.shape[-1]-1)
                node_distance_administrator = shortest_path_length(self.G, target=0)

                for node in self.G.nodes():
                    self.X[node][0] = node_distance_administrator[node]
                    self.X[node][1] = node_distance_instructor[node]
                    
            
            if distance==False:  
                self.X=np.matrix([
                    [-1.,1.]
                   for j in range(self.A.shape[0])
                    ])
        
        self.A=self.A+self.I
        self.feature_shape=self.X.shape[-1]
        self.model_inp=[self.X,self.A]
        self.A_hat=self.A*self.X
        
        self.D=np.array(np.sum(self.A,axis=0))[0]
        self.D=np.matrix(np.diag(self.D))
        self.F=(self.D**(-1)*self.A)
        
        
    def convert_tensor(self):
            
        self.A=k.constant(self.A)
        self.X=k.constant(self.X)
        self.A_hat=k.constant(self.A_hat)
        self.I=k.constant(self.I)
        self.D=k.constant(self.D)
        self.F=k.constant(self.F)
        
class ExperimentalGCN():
    def create_network(self,G,X,distance):
        
        self.network=Network()
        self.network.build(G,X,distance)
        self.network.convert_tensor()
        self.F= Input(shape=(self.network.F.shape[0],self.network.F.shape[1]))
        self.I=Input(shape=(self.network.I.shape[0],self.network.I.shape[1]))


    def GCN(self,inp,units,shape,features,activation):
    

        self.model=keras.Sequential()
        self.model.add(Input(tensor=inp))
        self.model.add(GraphNeuralNetwork(self.units,int(self.shape),keras.activations.relu,keras.initializers.glorot_uniform,None,keras.initializers.zeros,None,self.features))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(units=units,activation=activation))
        return self.model
    def extract_features(self,units,output_feature_shape,activation,optimizer,epochs,steps_p_epoch,pred_steps):

        self.inp=Input(tensor=(self.network.F))
        self.units=units
        self.shape=self.network.A_hat.shape[-1]
        self.output_feature_shape=output_feature_shape
        self.features=self.network.A_hat
        self.intermediate_activation=keras.activations.relu
        self.activation=activation
        self.epochs=epochs
        self.steps_p_epoch=steps_p_epoch
        self.pred_steps=pred_steps
        self.optimizer=optimizer
        self.model=self.GCN(self.network.F,self.units,self.shape,self.features,self.intermediate_activation)  
        self.model=self.GCN(self.network.F,self.output_feature_shape,self.model.layers[1].output.shape[-1],self.model.layers[1].output,keras.activations.sigmoid)    
        self.model.summary()
        self.model.compile(loss='binary_crossentropy',optimizer=self.optimizer,metrics=["accuracy"])
        self.model.fit(self.network.F,epochs=epochs,batch_size=self.network.F.shape[-1],steps_per_epoch=self.steps_p_epoch)
        predictions=self.model.predict(self.network.F,steps=self.pred_steps)
        return predictions
        
    def draw_graph(self,predictions,shape,node_size,with_labels,dim_1,dim_2):
        
       
        self.pos = {}
        self.shape=shape
        keys=range(predictions.shape[1])
        cols=range(predictions.shape[1])
        for i in keys:
            self.pos[i]=cols[i]
        self.colors = []
        self.idx=[]
        for v in range(shape):
            max_idx = predictions[v].argmax()
            self.cls = self.pos[max_idx]
            self.idx.append(predictions[v].argmax())
            self.colors.append(self.cls)
        nx.draw(self.network.G, node_color=self.colors,with_labels=with_labels, node_size=node_size)
        fig = plt.figure(figsize=(dim_1,dim_2))
        fig.clf()
        plt.show()
        return self.idx        
    def get_outcome(self,node):
        out_class=self.idx[node-1]
        assert node-1<self.shape
        return out_class
        
        
class feature_kernels():
    def centrality_kernel(self,centrality,Graph):
        if  Graph is not None:
            self.G=Graph
        
        if Graph is None:
            self.G= karate_club_graph()
        
        self.centrality=centrality
        if centrality is not None:
            self.node_centrality=centrality(self.G)
        
        if centrality is None:
            self.node_centrality=degree_centrality(self.G)
        
        self.X= np.matrix([
                [self.node_centrality[j]]
               for j in (self.G)
                ])
        return self.X
    def feature_random_weight_kernel(self,no_features,Graph):
        if  Graph is not None:
            self.G=Graph
        
        if Graph is None:
            self.G= karate_club_graph()
        
        self.X=np.matrix([
                 [np.random.randn() for i in range((no_features))]
                for j in range(self.G.number_of_nodes())
                 ])
        return self.X
    def feature_distributions(self,distribution,Graph):
        if  Graph is not None:
            self.G=Graph
        
        if Graph is None:
            self.G= karate_club_graph()
        self.distribution=distribution
        self.X=np.matrix([
                 [i for i in self.distribution]
                for j in range(self.G.number_of_nodes())
                 ])
        return self.X
        

        