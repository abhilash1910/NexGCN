# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 00:49:36 2020

@author: Abhilash
"""

from distutils.core import setup
setup(
  name = 'NexGCN',         
  packages = ['NexGCN'],   
  version = '0.2',       
  license='MIT',        
  description = 'A Spectral Sigmoid Graph Convolution Network Library for binary classification on Networkx Graphs compatible with Keras and Tensorflow',   
  author = 'ABHILASH MAJUMDER',
  author_email = 'debabhi1396@gmail.com',
  url = 'https://github.com/abhilash1910/NexGCN',   
  download_url = 'https://github.com/abhilash1910/NexGCN/archive/v_0.4.tar.gz/',    
  keywords = ['graph_convolution_network', 'binary classification', 'GCN','graph neural network','Spectral GCN','Networkx GCN','centrality GCN','keras GCN','sigmoid Graph convolution network'],   
  install_requires=[           

          'numpy',         
          'matplotlib',
          'keras',
          'tensorflow',
          'pandas',
          'networkx'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
