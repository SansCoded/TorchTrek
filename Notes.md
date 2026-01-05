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

----------------------------------------------------------------------------------------------------------------------------------------------------
**Autograd -**

**What is Autograd in PyTorch?**

In simple terms, Autograd is PyTorch’s way of automatically calculating the gradients (or derivatives) for you during training.
When we train a neural network, we want to update its weights (or parameters) to make it better at predictions. To do this, we need to know how the loss (error) changes when we tweak the weights. This is where gradients come in — they tell us how to adjust the weights.
Autograd is like a helper that tracks all the operations you do to tensors (PyTorch's version of arrays or matrices) and then, when you ask it, automatically calculates the gradients needed to update those tensors. It helps in speeding up the training process because it removes the need for you to manually compute all those gradients!

**Why Do We Need Autograd?**

When we train neural networks, we usually do something called backpropagation. Backpropagation is a technique that uses the gradients to update the weights in the right direction. Without autograd, we would need to manually compute these gradients, which is not only super complicated but also time-consuming. Autograd automates this, so we can focus on building and improving the model instead of worrying about the math behind it.
Example :-  Let’s go through a simple example to understand how autograd works in PyTorch:

-------------------------------------------------------------------------
import torch

#Create a tensor with requires_grad=True so PyTorch knows to track operations on it

**x = torch.tensor(2.0, requires_grad=True)**

#Perform some operations

**y = x ** 2 + 3 * x + 5  # y = x^2 + 3x + 5**

#Now, we want to calculate the derivative of y with respect to x, so we call .backward()

**y.backward(**)

#The gradient (dy/dx) is stored in x.grad

**print(x.grad)** # It will print the derivative of y = 2x + 3 at x = 2, which is 7

-----------------------------------------------------------------------
**Step-by-Step Explanation:**

Creating a Tensor: x = torch.tensor(2.0, requires_grad=True) — We create a tensor with the value 2.0 and set requires_grad=True, meaning PyTorch should keep track of any operations done to this tensor.

Performing Operations: y = x ** 2 + 3 * x + 5 — This is just a simple mathematical equation. PyTorch keeps track of the fact that y depends on x.

Calling .backward(): This is the key part. When we call y.backward(), PyTorch automatically calculates how y changes with respect to x (this is the gradient). In our example, the derivative of y with respect to x (dy/dx) is 2x + 3, so at x = 2.0, the gradient will be 7.0.

Accessing the Gradient: x.grad contains the gradient. In this case, it will print 7.0, because the derivative of x^2 + 3x + 5 at x = 2.0 is 7.0.


**Why Is This Important?**


- Efficiency: You don’t have to manually compute gradients.
- Learning: The gradients are used in the optimization process to adjust the weights and minimize the loss function (making the model better).
- Flexibility: You can perform complex operations and PyTorch will track all dependencies, giving you the correct gradients when needed.

In short, autograd in PyTorch is like a powerful calculator that helps you update your model’s weights automatically during training by calculating how changes to those weights affect the error (loss).
Does that make sense?
