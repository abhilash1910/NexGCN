# NexGCN- Networkx Graph Convolution Network

## A Spectral Sigmoid Graph Convolution Network Library for Networkx Graphs compatible with Keras/Tensorflow :robot:

This library is a modified binary semi supervised classification implementation of Semi Supervised Classification with Graph Convolution Network(Thomas Kipf et.al 2016) .Link:[https://arxiv.org/pdf/1609.02907v4.pdf]. This is compatible with Keras and Tensorflow (keras version >=2.0.6).
The library builds a semi supervised binary classification kernel by using networkx graphs as inputs along with features. Features like statistical centrality metrics, randomised distributions as well as weights are provided as additional kernels for evaluation. An example can be to classify the nodes of the Graph using Katz Centrality as the feature vector over the Deep GCN layer. Additional feature vectors can be provided manually as well.

## Dependencies

<a href="https://www.tensorflow.org/">Tensorflow</a>

<a href="https://keras.io/">Keras</a>

<a href="https://networkx.github.io/">NetworkX</a>

## Usability

The library or the Layer is compatible with Tensorflow and Keras. Installation is carried out using the pip command as follows:

```python
pip install NexGCN==0.1
```

For using inside the Jupyter Notebook or Python IDE (along with Keras layers):

```python
import NexGCN.NexGCN as venom
```
**Technical Aspects of GCN Library**


The library caan be used by passing a Networkx graph, and then providing the features. Any Networkx Graph can be used in this context; however there are restrictions in terms of the default features which are present in the library.Except MultiGraph (from Networkx), features which include centrality (degree,eigenvector,katz,betweenness,closeness) metrics can be used with any graphs (also with graphs converted from pandas dataframe). Features like randomised weight initialization or using a probabilistic distribution (poisson, beta,binomial,normal etc.) can be used for all types of Networkx Graphs. If no external feature vectors are provided, then a feature vector having alternate -1 and 1 are provided (if arguement in constructor is set to False).The major aim of this library is to reduce features and extract the nodes which are related to a particular class in a binary classification model. For any Networkx Graphs,having (m X n) feature vectors , where m is the number of nodes in the Graph and n is the feature size for each node, the library classifies the nodes to a  (m X 2) matrix. The latter represents under which class a particular node resides. The entire workflow is presented in the following steps, by using an Erdos Renyi Graph as an example:

**Importing necessary libraries**: For using this we have to import keras, tensorflow, networkx ,numpy,pandas and matplotlib as the major libraries.

```python
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
import NexGCN.NexGCN as venom
```

**Creating a Networkx Graph**: The first step is to create a Networkx Graph. This can be done by manually creating a graph, using the Graph generators in Networkx such as Zachary Karate Class,Erdos Renyi, Football, etc,   or by converting from knowledge graph corpora (Cora,Stanford Knowledge Graph having formats .txt,.list),pandas data frames.An example is shown here using Erdos Renyi Graph, and there are samples which use data from a pandas dataframe as well as Zachary Karate Club.

```python
Gr = nx.gnm_random_graph(70,140)
```

**Instantiating an Object of the NexGCN class**: When creating an Object of the NexGCN class, we have to keep in mind certain important functions.Since the user has to secify the hyperparameters for the deep GCN model,this provides an overview of some of the functions which will be used for interfacing. The entire library consists of 4 classes - 


The *Network* class responsible for graph creation,spectral feature extraction,and tensor conversion which converts the graph to be passed into the Deep GCN layers. 

The *GraphNeuralNetwork* class contains the actual Deep Learning layers for dual Graph convolution and also uses a systematic Relu activation in the intermediate stages. The final layer is a Dense MLP with sigmoid activation for binary classification. The kernels and biases are initialised by hyperparameters.


The *ExperimentalGCN* class is a wrapper which combines the *Network* class and *GraphNeuralNetwork* class. As of now the model supports Sequential layers from the Keras API The class also contains the necessary functions for plotting the finalized Graph, and presenting the outputs.


The *feature_kernels* is an additional class which contains implementations for including different functions for generating the feature vectors. These include centrality, randomised weight initializers, and distributions.

For our use case, we will just create an object of the *ExperimentalGCN* wrapper class and an object of the *feature_kernels* class as follows:
```python
Gr = nx.gnm_random_graph(70,140)
exp=venom.ExperimentalGCN()
kernel=venom.feature_kernels()
```

**Calling the GCN Layer**: This step involves creating a feature vector to compress and also to extract from. This is where the *feature_kernels* class is important. The shape of the feature is (m X n) where n is the number of features for each of the m nodes in the graph. The following sample shows the three functions inside the class, namely:

The *centrality_kernel* (takes as arguements, the centrality function and the Graph)for specifying features based on any centrality features.

The *feature_random_weight_kernel* (takes as arguements, the number of weights for each node and the Graph) for specifying the features based on weights.

The *feature_distributions* (takes as arguements the numpy distribution from numpy.random and the Graph) for specifying the features based on a distribution.

```python
#X=kernel.centrality_kernel(katz_centrality,Gr)
X=kernel.feature_random_weight_kernel(34,Gr)
#X=kernel.feature_distributions(np.random.poisson(4,9),Gr)
```

The *create_network*  (takes as arguements the Graph, the feature vector X, and an optional variable True/False)function is used for preprocessing the Graph before passing it to the Deep GCN layers. The third variable is also used for creating feature vectors if the previous 3 methods are not used or no external feature is provided. If set to False, it creates a  (m X 2) size feature vector having weights (-1,1) for each nodes in the graph. If set to True, then shortest distance between each node and the terminal points (0 and m-1 if there are m nodes) is taken as the feature vector . It is recommended to use the alternate feature kernels or external generators for creating the features if error is shown for shorted distance estimate (experimental). If all the three parameters are of form (None,None,True/False) , this implies Zachary Karate Club graph will be analyzed with either the unit weights or shorted distance as mentioned above.
```python
exp.create_network(Gr,X,True)
```

The next step is to make the predictions by passing through the *extract_binary_features* function under the *ExperimentalGCN* class. It takes as arguements - the kernel size,the output dimension (2 since binary), the activation (sigmoid recommended for more than 98% accuracy), the optimizer, the number of epochs, the steps per epoch and the verbose for keras training. The output is a a compressed output feature map of size (m X 2) which represents which class a particular node belongs to. 
```python
predictions=exp.extract_binary_features(2048,2,keras.activations.sigmoid,'adam',5,20,1)
print(predictions)
```

**Plotting the Class Graph and Extracting output class**: The final step is to render the nodes in a binary classification format with 2 distinct colors.The *draw_graph* method from *ExperimentalGCN* class takes as arguements the predictions from the previous *extract_binary_features* method, the number of nodes in the Graph (it is recommended to keep it as *network.F.shape[-1]*), the dpi of the image, a boolean whether to display the labels, the dimensions of the plot(x,y), and the 2 colors for the 2 classes. The output is a matplotlib figure of the binary classified Graph.
```python
exp.draw_graph(predictions,exp.network.F.shape[-1],300,True,90,90,'#00FFFF','#FF00FF')
```
The *get_outcome* method returns a value ,either 0 or 1 depending on the input node provided as arguement. If a particular node is of class 0, it outputs 0 and vice versa.
```python
output_class=exp.get_outcome(37)
```

The training steps almost reaches 99.8% percent accuracy and the screenshot is shown below:
<img src="https://github.com/abhilash1910/NexGCN/blob/master/Training.PNG">Training on NexGCN</img>

The training on Zachary Karate Club is present in the examples (Test,Test_networkx and Test_pandas python files):
<img src="https://github.com/abhilash1910/NexGCN/blob/master/Images/gcn_zakary1-katz_centrality.png">NexGCN for Binary Classification on Zachary Karate Club using Katz centrality</img>

## Further Development

To enhance the entire process and to increase the robustness of the Spectral Graph learning process, further modules will be added.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

MIT
