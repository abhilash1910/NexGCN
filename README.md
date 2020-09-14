# NexGCN- Networkx Graph Convolution Network

## A Spectral Sigmoid Graph Convolution Network Library for Networkx Graphs compatible with Keras/Tensorflow :robot:

This library is a modified binary semi supervised classification implementation of Semi Supervised Classification with Graph Convolution Network(Thomas Kipf et.al 2016) .Link:[https://arxiv.org/pdf/1609.02907v4.pdf]. This is compatible with Keras and Tensorflow (keras version >=2.0.6).
The library builds a semi supervised binary classification kernel by using networkx graphs as inputs along with features. Features like statistical centrality metrics, randomised distributions as well as weights are provided as additional kernels for evaluation. An example can be to classify the nodes of the Graph using Katz Centrality as the feature vector over the Deep GCN layer. Additional feature vectors can be provided manually as well.

## Usability

The library or the Layer is compatible with Tensorflow and Keras. Installation is carried out using the pip command as follows:

```python
pip install NexGCN==0.1
```

For using inside the Jupyter Notebook or Python IDE (along with Keras layers):

```python
import NexGCN.NexGCN as venom
```


## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

MIT
