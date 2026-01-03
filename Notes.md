# TorchTrek

**Introduction to PyTorch -**

PyTorch is an open-source deep learning framework developed by Facebook's AI Research lab (FAIR). It’s widely used for developing and training machine learning models, particularly in the fields of computer vision, natural language processing (NLP), and reinforcement learning.
PyTorch has become one of the most popular tools in the machine learning community because of its dynamic computational graph (also called eager execution), ease of use, and strong support from the research community

PyTorch uses Tensors as the primary data structure, which is similar to NumPy arrays but with the added capability to be run on GPUs, providing significant speedups for deep learning models. PyTorch supports various operations like automatic differentiation, GPU acceleration, and more, making it an excellent tool for building deep learning models.

---------------------------------------------------------------------------------------------------

**What is a Tensor?**

A Tensor is a multi-dimensional array that can store data of various types. It's a fundamental building block of PyTorch, allowing you to represent vectors, matrices, and higher-dimensional data. Tensors are the inputs, outputs, and parameters of neural networks.

In PyTorch, tensors are much like NumPy arrays, but they are more powerful because they can be moved between CPU and GPU (for faster computation). A tensor can be of any dimension:

- Scalar: A 0-dimensional tensor (just a single number).
- Vector: A 1-dimensional tensor (like a list of numbers).
- Matrix: A 2-dimensional tensor (like a grid of numbers).
- Higher-dimensional Tensors: 3 or more dimensions, like data used in deep learning models (e.g., images, videos, etc.).

**Why Tensors?**

- Efficient computations: Tensors support efficient computation on both CPUs and GPUs.
- Dynamic Graphs: Tensors help in creating dynamic computation graphs in deep learning models.
- Storage: Tensors can hold a variety of data types like integers, floating-point numbers, etc.

**Types of Tensors -**

1) Scalar (0D Tensor): A scalar tensor holds a single value.
Example: 5
Shape: ()

2) Vector (1D Tensor): A vector tensor is a one-dimensional array of values.
Example: [1, 2, 3, 4]
Shape: (4,)

3) Matrix (2D Tensor): A matrix tensor is a two-dimensional array (rows and columns).
Example:
[[1, 2, 3],
 [4, 5, 6]]
Shape: (2, 3)

4) 3D Tensor: A three-dimensional tensor could represent a stack of matrices.
Example:
[[[1, 2],
  [3, 4]],
 [[5, 6],
  [7, 8]]]
Shape: (2, 2, 2)

5) Higher-dimensional Tensors: These can represent more complex data like images, videos, or multi-dimensional time series data.

**Creating and Manipulating Tensors -**

PyTorch provides the following functions to create tensors:-

- torch.empty(): Returns an uninitialized tensor of a given size.

- torch.zeros(): Returns a tensor filled with zeros.

- torch.ones(): Returns a tensor filled with ones.

- torch.rand(): Returns a tensor with random values in the range [0, 1).


**Tensor Shapes and Reshaping -**

The shape of a tensor refers to the dimensions of the tensor, such as how many rows and columns it has. You can manipulate tensor shapes using various operations like reshape, flatten, unsqueeze, and squeeze.

- x.shape gives the shape of a tensor.
 
- .reshape() changes the shape of a tensor.

- .flatten() converts a tensor into a 1D tensor.

- .unsqueeze() adds a new dimension to a tensor.

- .squeeze() removes dimensions with size 1 from a tensor.

**Tensor Data Types -**

PyTorch supports various data types for tensors, such as float32, int64, float64, etc. Understanding the correct data type to use is essential for optimizing your models and ensuring that operations are performed correctly.

You can check the data type of a tensor using .dtype and convert the data type using .to().


  
