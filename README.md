# **PythonDeepLearning library (PDLib)** #

This is simple library for deep learning with Python. 
The library is based on the book "Grokking Deep Learning".
The library is distributed under the MIT license and can be downloaded and used by anyone.

----------


## How to install ##
To install, you can use the command:

    pip install PyDeepLib

Or download the repository from [GitHub](https://github.com/y-a-r-i-k/PDLib)

----------

## Using ##
In this file you will not find a detailed description and instructions for working with this library, only a description of each of the base classes of the library will be presented here.
If you want to know how this library works in depth or learn how to work with it to perfection, read Chapter 13 of the book "Grokking Deep Learning". The code for this library is very similar to the code in the book.

### Tensor ###
Initially, this library should have been called SimpleTensorLib, since Tensor is the main component of the library.
> A tensor is an abstract form of representation
> nested lists of numbers. A vector is a one-dimensional tensor. Matrix -
> two-dimensional tensor, and structures with a large number of dimensions are called
> n-dimensional tensors.
  
The **Tensor class** (located in the tensorclass.py file) is the main class in the library and has a number of functions:   
  
- *all_children_grads_accounted_for*() Checks if gradients are derived from all children;
  
- *backward*() Responsible for backpropagation logic;
  
- A set of functions responsible for backpropagation operations;
  
  
During initialization, the class takes an input parameters:
  
- *data* - The actual data. It must be a numpy array;
  
- *autograd* - Use automatic gradient or not. Must be a boolean (true or false);
  
- *creators* - The list of tensors used for create current tensor. Default - None;
  
- *creation_op* - Stores the operations used to create. Default - None;
  
- *id* - ID. Optional parameter. If you do not set it, it is assigned automatically;
  
  
### Layer classes ###
  
List of all layer classes available for use:
  
- *Layer* - The base class of the layer, some others are inherited from it.
  
- *SGD* - automatic optimization.
  
- *Sequential* - a layer containing another layer.
  
- *MSELoss* - Loss function layer.
  
- *Embedding* - Layer with vector representation.
  
- *Tanh* and *Sigmoid* - Activation functions.
  
- *CrossEntropyLoss* - Cross entropy layer.
  
- *RNNCell* - Recurrent layer.
  
- *LSTMCell* - Long short term memory cell.
  
  
### Additional functions ###

- *File* - contained in the file speedfiein.py and is a built-in fragment of the speedfile library. Allows you to work with files faster. Read more [here](https://github.com/y-a-r-i-k/SpeedFile-for-Python).
  
- *Logging* - Fast logging class. In developing.

 
----------


## From the developer ##

> Hello. If you noticed this project, it's very cool! I did it just for myself while reading "Grokking Deep Learning". I'm just learning both neural networks and Python, so maybe something could be done better. If you have great knowledge about neural networks and Python, and want to help the project, then go to [my website](https://y-a-r-i-k.github.io/), there you will find my contacts, you can help me in the project, I will be glad for any cooperation or help, thanks in advance. In the near future, I probably will not be able to support the project, as I am very busy with my studies.
  
  