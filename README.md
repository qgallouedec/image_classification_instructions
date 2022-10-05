# Tutorial 1 : Image Classification 

MOD 4.6 Deep Learning & Artificial Intelligence: an introduction


## Introduction

The objective of this tutorial is to write a complete image classification program in Python.
Two classification models will be successively developed and tested: k-nearest neighbors (KNN) and neural networks (NN).

### Before you start

In this tutorial we use Python 3.7 or higher. Make sure it is properly installed. Make sure `numpy` is installed.

We assume that `git` is installed, and that you are familiar with the basic `git` commands. (Optionnaly, you can use GitHub Desktop.)
We also assume that you have access to the [ECL GitLab](https://gitlab.ec-lyon.fr/). If necessary, please consult [this tutorial](https://gitlab.ec-lyon.fr/edelland/inf_tc2/-/blob/main/Tutoriel_gitlab/tutoriel_gitlab.md).


### Code style

Your code must follow the [PEP8 recommendations](https://peps.python.org/pep-0008/). To help you format your code properly, you can use [Black](https://black.readthedocs.io/en/stable/). To help you sort your imports, you can use [isort](https://pycqa.github.io/isort/).


### Docstring

Your code must be properly documented. It must follow the [PEP257 recommendations](https://peps.python.org/pep-0257/). To help you document your code properly, you can use [pydocstyle](http://www.pydocstyle.org/en/stable/).


### Prepare your directory

1. Create a new blank project on the ECL GitLab (`New project` then `Create blank project`).
2. Fill in the form as follows.
   - Project name: `Image classification`.
   - Project slug: `image-classification`.
   - Visibility Level: public
   - Project Configuration: Initialize repository with a README
3. Clone the repository.
    ```bash
    git clone https://gitlab.ec-lyon.fr/<user>/image-classification
    ```
4. In this tutorial you will use files that should not be pushed to the remote repository. To ignore them when committing, you can put their path in a file named `.gitignore`. For simplicity, we use the [`.gitignore` file](https://github.com/github/gitignore/blob/main/Python.gitignore) recommended by GitHub for Python projects.


## Prepare the CIFAR dataset

The image database used for the experiments is CIFAR-10 which consists of 60 000 color images of size 32x32 divided into 10 classes (plane, car, bird, cat, ...).
This database can be obtained at the address https://www.cs.toronto.edu/~kriz/cifar.html where are also given the indications to read the data.

1. Create a folder named `data` in which you move the downloaded `cifar-10-batches-py` folder. Make sure that the `data` folder is ignored when commiting.
2. Create a Python file named `read_cifar.py`. Write the function `read_cifar_batch` taking as parameter the path of a single batch as a string, and returning:
      - a matrix `data` of size (`batch_size` x `data_size`) where `batch_size` is the number of available data in the batch, and `data_size` the dimension of these data (number of numerical values describing the data), and
      - a vector `labels` of size `batch_size` whose values correspond to the class code of the data of the same index in `data`.
    `data` must be `np.float32` array and `labels` must be `np.int64` array.
3. Write the function `read_cifar` taking as parameter the path of the directory containing the six batches (five `data_batch` and one `test_batch`) as a string, and returning
      - a matrix `data` of shape (`batch_size` x `data_size`) where `batch_size` is the number of available data in all batches (including `test_batch`), and
      - a vector `labels` of size `batch_size` whose values correspond to the class code of the data of the same index in `data`.
    `data` must be `np.float32` array and `labels` must be `np.int64` array.
4. Write the function `split_dataset` which splits the dataset into a training set and a test set. The data must be shuffled, so that two successive calls shouldn't give the same output. This function takes as parameter
      - `data` and `labels`, two arrays that have the same size in the first dimension.
      - `split`, a float between 0 and 1 which determines the split factor of the training set with respect to the test set.
    This function must return
      - `data_train` the training data,
      - `labels_train` the corresponding labels,
      - `data_test` the testing data, and
      - `labels_test` the corresponding labels.

## k-nearest neighbors

1. Create a Python fil named `knn.py`. Write the function `distance_matrix` taking as parameters two matrices and returns `dists`, the L2 Euclidean distance matrix. The computation must be done only with matrix manipulation (no loops).
    Hint: $`(a-b)^2 = a^2 + b^2 - 2 ab`$
2. Write the function `knn_predict` taking as parameters:
      - `dists` the distance matrix between the train set and the test set,
      - `labels_train` the training labels, and
      - `k` the number of of neighbors.
    This function must return the predicted labels for the elements of `data_train`.

    **Note:** if the memory occupation is too important, you can use several batches for the calculation of the distance matrix (loop on sub-batches of test data).
3. Write the function `evaluate_knn` taking as parameters:
      - `data_train` the training data,
      - `labels_train` the corresponding labels,
      - `data_test` the testing data,
      - `labels_test` the corresponding labels, and
      - `k` the number of of neighbors.
    This function must return the classification rate (accuracy).
4. For `split_factor=0.9`, plot the variation of the accuracy as a function of `k` (from 1 to 20). Save the plot as an image named `knn.png` in the directory `results`.

## Artificial Neural Network

The objective here is to develop a classifier based on a multilayer perceptron (MLP) neural network.

First of all, let's focus on the backpropagation of the gradient with an example. If you still have trouble understanding the intuition behind the back propagation of the gradient, check out this video: [3Blue1Brown/Backpropagation calculus | Chapter 4, Deep learning](https://www.youtube.com/watch?v=tIeHLnjs5U8).


The weight matrix of the layer $`L`$ is denoted $`W^{(L)}`$. The bias vector of the layer $`L`$ is denoted $`B^{(L)}`$. We choose the sigmoid function, denoted $`\sigma`$, as the activation function. The output vector of the layer $`L`$ before activation is denoted $`Z^{(L)}`$. The output vector of the layer $`L`$ after activation is denoted $`A^{(L)}`$. By convention, we note $`A^{(0)}`$ the network input vector. Thus $`Z^{(L+1)} = W^{(L+1)}A^{(L)} + B^{(L+1)}`$ and $`A^{(L+1)} = \sigma\left(Z^{(L+1)}\right)`$. Let's consider a network with one hidden layer. Thus, the output is $`\hat{Y} = A^{(2)}`$.
Let $`Y`$ be the labels (desired output). We use mean squared error (MSE) as the cost function. Thus, the cost is $`C = \frac{1}{N_{out}}\sum_{i=1}^{N_{out}} (\hat{y_i} - y_i)^2`$.

1. Prove that $`\sigma' = \sigma \times (1-\sigma)`$
2. Express $`\frac{\partial C}{\partial A^{(2)}}`$, i.e. the vector of $`\frac{\partial C}{\partial a^{(2)}_i}`$ as a function of $`A^{(2)}`$ and $`Y`$.
3. Using the chaining rule, express $`\frac{\partial C}{\partial Z^{(2)}}`$, i.e. the vector of $`\frac{\partial C}{\partial z^{(2)}_i}`$ as a function of $`\frac{\partial C}{\partial A^{(2)}}`$ and $`A^{(2)}`$.
4. Similarly, express $`\frac{\partial C}{\partial W^{(2)}}`$, i.e. the matrix of $`\frac{\partial C}{\partial w^{(2)}_{i,j}}`$ as a function of $`\frac{\partial C}{\partial Z^{(2)}}`$ and $`A^{(1)}`$.
5. Similarly, express $`\frac{\partial C}{\partial B^{(2)}}`$ as a function of $`\frac{\partial C}{\partial Z^{(2)}}`$.
6. Similarly, express $`\frac{\partial C}{\partial A^{(1)}}`$ as a function of $`\frac{\partial C}{\partial Z^{(2)}}`$ and $`W^{(2)}`$.
7. Similarly, express $`\frac{\partial C}{\partial Z^{(1)}}`$ as a function of $`\frac{\partial C}{\partial A^{(1)}}`$ and $`A^{(1)}`$.
8. Similarly, express $`\frac{\partial C}{\partial W^{(1)}}`$ as a function of $`\frac{\partial C}{\partial Z^{(1)}}`$ and $`A^{(0)}`$.
9. Similarly, express $`\frac{\partial C}{\partial B^{(1)}}`$ as a function of $`\frac{\partial C}{\partial Z^{(1)}}`$.

Below is a Python code performing a forward pass and computing the cost in a network containing a hidden layer and using the sigmoid function as the activation function:

```python
import numpy as np

N = 30  # number of input data
d_in = 3  # input dimension
d_h = 3  # number of neurons in the hidden layer
d_out = 2  # output dimension (number of neurons of the output layer)

# Random initialization of the network weights and biaises
w1 = 2 * np.random.rand(d_in, d_h) - 1  # first layer weights
b1 = np.zeros((1, d_h))  # first layer biaises
w2 = 2 * np.random.rand(d_h, d_out) - 1  # second layer weights
b2 = np.zeros((1, d_out))  # second layer biaises

data = np.random.rand(N, d_in)  # create a random data
labels = np.random.rand(N, d_out)  # create a random labels

# Forward pass
a0 = data # the data are the input of the first layer
z1 = np.matmul(a0, w1) + b1  # input of the hidden layer
a1 = 1 / (1 + np.exp(-z1))  # output of the hidden layer (sigmoid activation function)
z2 = np.matmul(a1, w2) + b2  # input of the output layer
a2 = 1 / (1 + np.exp(-z2))  # output of the output layer (sigmoid activation function)
labels_pred = a2  # the predicted values are the outputs of the output layer

# Compute loss (MSE)
loss = np.mean(np.square(labels_pred - labels))
print(loss)
```

10. Create a Python file named `mlp.py`. Use the above code to write the function `learn_once_mse` taking as parameters:
      - `w1`, `b1`, `w2` and `b2` the weights and biases of the network,
      - `data` a matrix of shape (`batch_size` x `d_in`),
      - `labels` a matrix of shape (`batch_size` x `d_out`),
      - `learning_rate` the learning rate,

    that perform one gradient descent step, and returns:
      - `w1`, `b1`, `w2` and `b2` the updated weights and biases of the network,
      - `loss` the loss, for monitoring purpose.

For classification task, we prefer to use a binary cross-entropy loss. We also want to replace the last activation layer of the network with a softmax layer.

10. Write the function `one_hot` taking a (n)-D array as parameters and returning the corresponding (n+1)-D one-hot matrix.
11. Write a function `learn_once_cross_entropy` taking the the same parameters as `learn_once_mse` and returns the same outputs. The function must use a cross entropy loss and the last layer of the network must be a softmax. We admit that $`\frac{\partial C}{\partial Z^{(2)}} = A^{(2)} - Y`$. Where $`Y`$ is a one-hot vector encoding the label.
12. Write the function `evaluate_mlp` taking as parameter:
      - `data_train`, `labels_train`, `data_test`, `labels_test`, the training and testing data,
      - `d_h` the number of neurons in the hidden layer
      - `learning_rate` the learning rate, and
      - `num_epoch` the number of training epoch,

    that train an MLP classifier and return the test accuracy computed on the test set.
13. For `split_factor=0.9`, `d_h=64`, `learning_rate=0.1` and `num_epoch=10_000`, plot the evolution of accuracy across learning epochs. Save the graph as an image named `mlp.png` in the `results` directory.

## To be handed in

This work (KNN and MLP) must be done individually. The expected output is the archive containing the complete, minimal and functional code corresponding to the tutorial on https://gitlab.ec-lyon.fr.
To see the details of the expected, see the Evaluation section.

The last commit is inteded before Monday, November 16, 2022.


## To go further

### Unittest

Your code should contain unit tests. All unit tests should be contained in the `tests` directory located at the root of the directory.
We choose to use [pytest](https://docs.pytest.org/en/7.1.x/). To help you write unit tests, you can consult the pytest documentation.

### License

Your project should be properly licensed. Since it is your project, it is up to you to choose your license. In general, the license consists of a file named LICENSE in the root directory. A useful resource to help you choose: https://choosealicense.com/

### Deep dive into the classifier

Experiments will have to be carried out by studying the following variations:
- use image representation by descriptors (LBP, HOG, ...) instead of raw pixels using the `scikit-image` module.
- use of N-fold cross-validation instead of a fixed learning and testing subset.


## Evaluation

In this section, we present all the items on which the work is evaluated.

- ( /1) The function `read_cifar_batch` works as described
- ( /1) The function `read_cifar` works as described
- ( /1) The `split_dataset` works as described
- ( /1) The function `distance_matrix` works as described
- ( /1) The function `knn_predict` works as described
- ( /1) The graph `knn.png` shows the results obtained
- ( /3) Demonstrations of back propagation are done without error.
- ( /1) The function `learn_once_mse` works as described
- ( /1) The function `one_hot` works as described
- ( /1) The function `learn_once_cross_entropy` works as described
- ( /1) The function `evaluate_mlp` works as described
- ( /1) The graph `mlp.png` shows the results obtained
- ( /3) Unitest coverage
- ( /2) The guidlines about the project structure are all followed

    To check if the project has the right structure, install `tree` and run from the project directory:

    ```bash
    $ tree -I 'env|*__pycache__*'
    .
    └── tests
        └── test_knn.py

    1 directory, 1 file
    ```
    The output must strictly match the one provided above.
- ( /1) Project has a license
- ( /2) All functions are documented
- ( /1) All functions are documented and follow the pydocstyle
- ( /1) The code is properly formatted

    To check if the code is properly formatted, install [Black](https://github.com/psf/black) and run from the project repository:

    ```bash
    $ black --check . --exclude env
    ```

    ```bash
    $ isort --check . -s env
    ```

    These two tests must pass without error.
