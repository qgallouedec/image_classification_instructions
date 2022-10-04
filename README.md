# Image Classification Instructions

## Introduction

The objective of this tutorial is to write a complete image classification program in Python. 
Two classification models will be successively developed and tested: k-nearest neighbors (KNN) and neural networks (NN).

### Before you start

In this tutorial we use Python 3.9.14. Make sure you have this version of Python installed.

```bash
% python3.9 --version
Python 3.9.14
```

We assume you are familiar with the `venv` module of Python and with basic `git` commands.
We assume that you have access to the [ECL GitLab](https://gitlab.ec-lyon.fr/).

### Prepare your directory

1. Connect to https://gitlab.ec-lyon.fr.
2. Create a new blank project (`New project` then `Create blank project`).
3. Fill in the form as follows.
   - Project name: `Image classification`.
   - Project slug: `image-classification`.
   - Visibility Level: public
   - Project Configuration: Initialize repository with a README
4. Clone the repository.
   
```bash
git clone https://gitlab.ec-lyon.fr/<user>/image-classification
```

### Prepare the Python envrionment

1. In the project direcotry, create a virtual environment.

```bash
python3.9 -m venv env
```

2. Source the envrionement.

```bash
source env/bin/activate
```

3. Upgrade `pip`.

```bash
pip install --upgrade pip
```

4. The environment files should not be pushed to the remote directory. To have these files ignored when committing, create a `.gitignore` file containing `env`. Similarly, we want to ignore Python cache file, thus add `__pycache__` to `.gitignore`.
5. In this project, we use `numpy` package for matrices manipulation and the `scikit-image` package for image manipulation. Thus, create a requirement file named `requirements.txt` containing:

```txt
numpy
scikit-image
```

6. Install the above mentioned dependencies.

```bash
pip install -r requirements.txt
```


## Prepare the CIFAR dataset

The image database used for the experiments is CIFAR-10 which consists of 60 000 color images of size 32x32 divided into 10 classes (plane, car, bird, cat, ...).
This database can be obtained at the address https://www.cs.toronto.edu/~kriz/cifar.html where are also given the indications to read the data.

1. Create a folder named `data` in which you move the downloaded `cifar-10-batches-py` folder. Make sure that the `data` folder is ignored when commiting.
2. Create a Python file named `read_cifar.py`. Write the function `read_cifar_batch` taking as parameter the path of a single batch as a string, and returning:
      - a matrix `data` of size (`batch_size` x `data_size`) where `batch_size` is the number of available data in the batch, and `data_size` the dimension of these data (number of numerical values describing the data), and
      - a vector `labels` of size `batch_size` whose values correspond to the class code of the data of the same index in `data`. 
    `data` and `labels` must be `np.float32` arrays.
3. Write the function `read_cifar` taking as parameter the path of the directory containing the six batches (five `data_batch` and one `test_batch`) as a string, and returning 
      - a matrix `data` of shape (`batch_size` x `data_size`) where `batch_size` is the number of available data in all batches (including `test_batch`), and
      - a vector `labels` of size `batch_size` whose values correspond to the class code of the data of the same index in `data`. 
    `data` and `labels` must be `np.float32` arrays.
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
3. Write the function `evaluate_knn` taking as parameters:
      - `data_train` the training data,
      - `labels_train` the corresponding labels,
      - `data_test` the testing data,
      - `labels_test` the corresponding labels, and
      - `k` the number of of neighbors.
    This function must return the classification rate (accuracy).
4. For `split_factor=0.9`, plot the variation of the accuracy as a function of `k` (from 1 to 20). Save the plot as an image under the directory `results`.
5. For `split_factor=0.9`, plot the variation of the accuracy as a function of `k` (from 1 to 20). Save the plot as an image in the directory `results`.

## Artificial Neural Network

The objective here is to develop a classifier based on a multilayer perceptron (MLP) neural network.

First of all, let's focus on the backpropagation of the gradient with an example.
Let's consider a network with a hidden layer.

The weight matrix of the layer $`L`$ is denoted $`W^{(L)}`$. The bias vector of the layer $`L`$ is denoted $`B^{(L)}`$. We choose the sigmoid function, denoted $`\sigma`$, as the activation function. The output vector of the layer $`L`$ before activation is denoted $`Z^{(L)}`$. The output vector of the layer $`L`$ after activation is denoted $`A^{(L)}`$. By convention, we note $`A^{(0)}`$ the network input vector. Thus $`Z^{(L+1)} = W^{(L+1)}A^{(L)} + B^{(L+1)}`$ and $`A^{(L+1)} = \sigma\left(Z^{(L+1)}\right)`$. In our example, the output is $`\hat{Y} = A^{(2)}`$.
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
      - `learning_rate` the learning rate,
      - `num_epoch` the number of training epoch
    
    that train an MLP classifier and return the test accuracy computed on the test set.


## To be handed in

This work (KNN and NN) must be done individually. The expected output is the archive containing the complete, minimal and functional code corresponding to the tutorial on https://gitlab.ec-lyon.fr.
To see the details of the expected, see the Evaluation section.

The last commit is inteded before Monday, November 16, 2022.


## Additional requirements

### Unittest

Your code must contain unit tests. All unit tests should be contained in the `tests` directory located at the root of the directory.
We choose to use [pytest](https://docs.pytest.org/en/7.1.x/). To help you write unit tests, you can consult the pytest documentation.


### Code style

Your code must strictly follow the [PEP8 recommendations](https://peps.python.org/pep-0008/). To help you format your code properly, you can use [Black](https://black.readthedocs.io/en/stable/). To help you sort your imports, you and [isort](https://pycqa.github.io/isort/)


### Docstring

Your code must be properly documented. It must follow the [PEP257 recommendations](https://peps.python.org/pep-0257/). To help you document your code properly, you can use [pydocstyle](http://www.pydocstyle.org/en/stable/).

### License

Your project must be properly licensed. Since it is your project, it is up to you to choose your license. In general, the license consists of a file named LICENSE in the root directory. A useful resource to help you choose: https://choosealicense.com/


## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.


## Evaluation

Dans ce document, j'explique tous les point de controle que je vais regarder. Je peux donner une aide sur comment obtenir rapidement le résultat. La structure est

- [ ] The project ...
    To check if it is correct you can do ...

- [ ] The project has the right structure.

    To check if the project has the right structure, install `tree` and run from the project directory:

    ```bash
    $ tree -I 'env|*__pycache__*'
    .
    └── tests
        └── test_knn.py

    1 directory, 1 file
    ```

    The output must strictly match the one provided above. 

- [ ] The project is properly formatted.

    To check if the code is properly formatted, install [Black](https://github.com/psf/black) and run from the project repository:

    ```bash
    $ black --check . --exclude env
    ```

    ```bash
    $ isort --check . -s env
    ```

    These two tests must pass without error. 

- [ ] The project is properly documented.


- [ ] The project is properly licensed.
- [ ] All the unit test pass
- [ ] The project has good coverage.