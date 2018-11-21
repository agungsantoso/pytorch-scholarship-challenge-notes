# PyTorch Scholarship Challenge 2018/2019 Notes


![GitHub](https://img.shields.io/github/license/mashape/apistatus.svg)

<p align="center">
  <img src="./images/pytorch_scholarship.png" width="25%">
</p>

A collection of notes on PyTorch Scholarship Challenge 2018/2019.

Contributions are always welcome!

<!-- toc -->

- [AMA](#ama)
- [Lesson 2](#lesson-2)
  * [Lectures](#lectures)
    + [Classification Problems](#classification-problems)
    + [Decision Boundary](#decision-boundary)
    + [Perceptrons](#perceptrons)
    + [Why "Neural Networks"?](#why-neural-networks)
    + [Perceptrons as Logical Operators](#perceptrons-as-logical-operators)
    + [Perceptron Trick](#perceptron-trick)
    + [Perceptron Algorithm](#perceptron-algorithm)
    + [Non-Linear Regions](#non-linear-regions)
    + [Error Functions](#error-functions)
    + [Log-loss Error Function](#log-loss-error-function)
    + [Discrete vs Continous](#discrete-vs-continous)
    + [Softmax](#softmax)
    + [One-Hot Encoding](#one-hot-encoding)
    + [Maximum Likelihood](#maximum-likelihood)
    + [Cross-Entropy](#cross-entropy)
    + [Multi-Class Cross Entropy](#multi-class-cross-entropy)
    + [Logistic Regression](#logistic-regression)
    + [Gradient Descent](#gradient-descent)
    + [Feedforward](#feedforward)
    + [Backpropagation](#backpropagation)
    + [Overfitting & Underfitting](#overfitting--underfitting)
      - [Overfitting](#overfitting)
      - [Underfitting](#underfitting)
    + [Early Stopping](#early-stopping)
    + [Regularization](#regularization)
    + [Dropout](#dropout)
    + [Local Minima](#local-minima)
    + [Random Restart](#random-restart)
    + [Momentum](#momentum)
  * [Quizes](#quizes)
  * [Notebooks](#notebooks)
- [Lesson 3](#lesson-3)
  * [Interview](#interview)
    + [Origins of PyTorch](#origins-of-pytorch)
    + [Debugging and Designing PyTorch](#debugging-and-designing-pytorch)
    + [From Research to Production](#from-research-to-production)
    + [Hybrid Frontend](#hybrid-frontend)
    + [Cutting-edge Applications in PyTorch](#cutting-edge-applications-in-pytorch)
    + [User Needs and Adding Features](#user-needs-and-adding-features)
    + [PyTorch and the Facebook Product](#pytorch-and-the-facebook-product)
    + [The Future of PyTorch](#the-future-of-pytorch)
    + [Learning More in AI](#learning-more-in-ai)
- [Lesson 4](#lesson-4)
  * [Lectures](#lectures-1)
    + [Single layer neural networks](#single-layer-neural-networks)
    + [Networks Using Matrix Multiplication](#networks-using-matrix-multiplication)
    + [Neural Networks in PyTorch](#neural-networks-in-pytorch)
    + [Network Architectures in PyTorch](#network-architectures-in-pytorch)
    + [Classifying Fashion-MNIST](#classifying-fashion-mnist)
    + [Inference and Validation](#inference-and-validation)
    + [Saving and Loading Models](#saving-and-loading-models)
    + [Loading Image Data](#loading-image-data)
    + [Transfer Learning](#transfer-learning)
    + [Tips, Tricks, and Other Notes](#tips-tricks-and-other-notes)
  * [Quizes](#quizes-1)
  * [Notebooks](#notebooks-1)
- [Lesson 5](#lesson-5)
- [Lesson 6](#lesson-6)
- [Lesson 7](#lesson-7)
- [Lesson 8](#lesson-8)
- [Lesson 9](#lesson-9)
- [Challenge Project](#challenge-project)
- [Credits](#credits)

<!-- tocstop -->

## AMA

* [2018-11-09 A](https://drive.google.com/open?id=1eqQ2auQ4ClI_v1Rqm--cI2Vm5Ew5-F1x)
* [2018-11-09 B](https://drive.google.com/open?id=1er8e0DRRP61ugEysRPBLRRw_0E2kNwQA)
* [2018-11-13](https://drive.google.com/open?id=13cp-IkrGet6mb6dzi55bBTLAWuP6S4zw)
* [2018-11-14](https://drive.google.com/open?id=1fGZe2yHuq3_hAdK4ZhodfyoNk5ofZq_j)
* [2018-11-15 A](https://drive.google.com/open?id=1x-QXNcVXKu-VokKvRlX9NreT9atQTyHK)
* [2018-11-15 B](https://drive.google.com/open?id=1mstM3SvvhIIwcBbtClzXaLx37Z9oHozX)

## Lesson 2
### Lectures
#### Classification Problems
  The problem of identifying to which of a set of categories (sub-populations) a new observation belongs.

  <p align="center">
    <img src="./images/lesson-2/classification-problems.PNG" width="50%">
  </p>

#### Decision Boundary
  The separator between classes learned by a model in a binary class or multi-class classification problems. For example, in the following image representing a binary classification problem, the decision boundary is the frontier between the blue class and the red class:

  * Linear Boundaries
    <p align="center">
      <img src="./images/lesson-2/linear-boundaries.PNG" width="50%">
    </p>

  * Higher Dimensions
    <p align="center">
      <img src="./images/lesson-2/higher-dimensions.PNG" width="50%">
    </p>

#### Perceptrons
  A system (either hardware or software) that takes in one or more input values, runs a function on the weighted sum of the inputs, and computes a single output value. In machine learning, the function is typically nonlinear, such as ReLU, sigmoid, or tanh.

  In the following illustration, the perceptron takes n inputs, each of which is itself modified by a weight before entering the perceptron:

  <p align="center">
    <img src="./images/lesson-2/perceptrons.PNG" width="50%">
  </p>

  A perceptron that takes in n inputs, each multiplied by separate
  weights. The perceptron outputs a single value.

  Perceptrons are the (nodes) in deep neural networks. That is, a deep neural network consists of multiple connected perceptrons, plus a backpropagation algorithm to introduce feedback.

#### Why "Neural Networks"?

  <p align="center">
    <img src="./images/lesson-2/why-neural-network.PNG" width="50%">
  </p>

#### Perceptrons as Logical Operators

* AND Perceptron

  <p align="center">
    <img src="./images/lesson-2/and-quiz.png" width="50%">
  </p>

* OR Perceptron

  <p align="center">
    <img src="./images/lesson-2/or-quiz.png" width="50%">
  </p>

    <p align="center">
    <img src="./images/lesson-2/and-to-or.png" width="50%">
  </p>

* NOT Perceptron
  Unlike the other perceptrons we looked at, the NOT operation only cares about one input. The operation returns a 0 if the input is 1 and a 1 if it's a 0. The other inputs to the perceptron are ignored.

* XOR Perceptron

  <p align="center">
    <img src="./images/lesson-2/xor.png" width="50%">
  </p>

#### Perceptron Trick
<p align="center">
  <img src="./images/lesson-2/perceptron-trick.PNG" width="50%">
</p>

#### Perceptron Algorithm
<p align="center">
  <img src="./images/lesson-2/perceptron-algorithm.PNG" width="50%">
</p>

#### Non-Linear Regions
<p align="center">
  <img src="./images/lesson-2/non-linear-regions.PNG" width="50%">
</p>

#### Error Functions
<p align="center">
  <img src="./images/lesson-2/error-functions.PNG" width="50%">
</p>

#### Log-loss Error Function
<p align="center">
  <img src="./images/lesson-2/log-loss-error-function.PNG" width="50%">
</p>

#### Discrete vs Continous
<p align="center">
  <img src="./images/lesson-2/discrete-vs-continous.PNG">
</p>

#### Softmax
A function that provides probabilities for each possible class in a multi-class classification model. The probabilities add up to exactly 1.0. For example, softmax might determine that the probability of a particular image being a duck at 0.67, a beaver at 0.33, and a walrus at 0. (Also called full softmax.)

<p align="center">
  <img src="./images/lesson-2/softmax.PNG" width="50%">
</p>

#### One-Hot Encoding

A sparse vector in which:

* One element is set to 1.
* All other elements are set to 0.

One-hot encoding is commonly used to represent strings or identifiers that have a finite set of possible values. For example, suppose a given botany data set chronicles 15,000 different species, each denoted with a unique string identifier. As part of feature engineering, you'll probably encode those string identifiers as one-hot vectors in which the vector has a size of 15,000.

<p align="center">
  <img src="./images/lesson-2/one-hot-encoding.PNG" width="50%">
</p>


#### Maximum Likelihood
<p align="center">
  <img src="./images/lesson-2/maximum-likelihood.PNG" width="50%">
</p>

#### Cross-Entropy

A generalization of Log Loss to multi-class classification problems. Cross-entropy quantifies the difference between two probability distributions.

<p align="center">
  <img src="./images/lesson-2/cross-entropy.PNG" width="50%">
</p>

#### Multi-Class Cross Entropy
<p align="center">
  <img src="./images/lesson-2/multi-class-cross-entropy.PNG" width="50%">
</p>

#### Logistic Regression
A model that generates a probability for each possible discrete label value in classification problems by applying a sigmoid function to a linear prediction. Although logistic regression is often used in binary classification problems, it can also be used in multi-class classification problems (where it becomes called multi-class logistic regression or multinomial regression).
<p align="center">
  <img src="./images/lesson-2/logistic-regresssion.PNG" width="50%">
</p>

#### Gradient Descent
A technique to minimize loss by computing the gradients of loss with respect to the model's parameters, conditioned on training data. Informally, gradient descent iteratively adjusts parameters, gradually finding the best combination of weights and bias to minimize loss.
<p align="center">
  <img src="./images/lesson-2/gradient-descent.PNG" width="50%">
</p>

#### Feedforward
<p align="center">
  <img src="./images/lesson-2/feedforward.PNG" width="50%">
</p>

#### Backpropagation
The primary algorithm for performing gradient descent on neural networks. First, the output values of each node are calculated (and cached) in a forward pass. Then, the partial derivative of the error with respect to each parameter is calculated in a backward pass through the graph.
<p align="center">
  <img src="./images/lesson-2/backpropagration.PNG" width="50%">
</p>

#### Overfitting & Underfitting
##### Overfitting
Creating a model that matches the training data so closely that the model fails to make correct predictions on new data.

##### Underfitting
Producing a model with poor predictive ability because the model hasn't captured the complexity of the training data. Many problems can cause underfitting, including:

* Training on the wrong set of features.
* Training for too few epochs or at too low a learning rate.
* Training with too high a regularization rate.
* Providing too few hidden layers in a deep neural network.

<p align="center">
  <img src="./images/lesson-2/overfitting-and-underfitting.PNG" width="50%">
</p>

#### Early Stopping
A method for regularization that involves ending model training before training loss finishes decreasing. In early stopping, you end model training when the loss on a validation data set starts to increase, that is, when generalization performance worsens.
<p align="center">
  <img src="./images/lesson-2/early-stopping.PNG" width="50%">
</p>

#### Regularization
The penalty on a model's complexity. Regularization helps prevent overfitting. Different kinds of regularization include:

* L1 regularization
* L2 regularization
* dropout regularization
* early stopping (this is not a formal regularization method, but can effectively limit overfitting)

<p align="center">
  <img src="./images/lesson-2/regularization.PNG" width="50%">
</p>

#### Dropout
A form of regularization useful in training neural networks. Dropout regularization works by removing a random selection of a fixed number of the units in a network layer for a single gradient step. The more units dropped out, the stronger the regularization. This is analogous to training the network to emulate an exponentially large ensemble of smaller networks.
<p align="center">
  <img src="./images/lesson-2/dropout.PNG" width="50%">
</p>

#### Local Minima
<p align="center">
  <img src="./images/lesson-2/local-minima.PNG" width="50%">
</p>

#### Random Restart
<p align="center">
  <img src="./images/lesson-2/random-restart.PNG" width="50%">
</p>

#### Momentum
A sophisticated gradient descent algorithm in which a learning step depends not only on the derivative in the current step, but also on the derivatives of the step(s) that immediately preceded it. Momentum involves computing an exponentially weighted moving average of the gradients over time, analogous to momentum in physics. Momentum sometimes prevents learning from getting stuck in local minima.
<p align="center">
  <img src="./images/lesson-2/momentum.PNG" width="50%">
</p>

### Quizes
* [Centurion's Note](https://files.slack.com/files-pri/TDBKE3X9D-FE3CP0FNW/download/udacity-pytorch-lesson2-notes.pdf)

### Notebooks
* [Gradient Descent](https://github.com/agungsantoso/deep-learning-v2-pytorch/blob/master/intro-neural-networks/gradient-descent/GradientDescent.ipynb)
* [Analyzing Student Data](https://github.com/agungsantoso/deep-learning-v2-pytorch/blob/master/intro-neural-networks/student-admissions/StudentAdmissions.ipynb)

## Lesson 3
### Interview
#### Origins of PyTorch
*  Soumith Chintala always wanted to be a visual effects artist at least when he started his undergrad and then he interned at a place and they said he's not good enough
* He was good at programming since he was a kid
* He try to find the next most magical thing and that was computer vision
* He had to find a professor in India which is really hard to afford who's doing this kind of stuff and it was just like one or two and He spent six months with the professor's lab
* He started picking up some things then went to CMU tried his hand at robotics and then finally landed at NYU and Yann LeCun's lab doing deep learning
* He got to NYU, he've been working on building up tooling.
* He worked on this project called EB learn which was like two generations before in terms of deep learning
* Then came around torch which is written by a few people
* He started getting pretty active and helping people out using torch and then developing a torch
* At some point we decided that we needed a new tool because all the as the field moves
* I went about building PyTorch mostly because we had a really stressful project that was really large and hard to build
* We started with just three of us and then we got other people interested
* About eight or nine people joined in part-time just adding feature and then slowly and steadily we started giving access to other people
* every week we would give access to about like ten people
* and then in Jan be released by doors to the public

#### Debugging and Designing PyTorch
* if you have a non-contiguous tensor and sent it through a linear layer it will just give you garbage
* a trade-off there where the readability comes at a cost of it being a little bit slow
* it should be very imperative very usable very pythonic but at the same time as fast as any other framework
* the consequences of that was like large parts of PyTorch live in C++ except whatever is user-facing
* you can attach your debugger you can print, those are still very very hackable

#### From Research to Production
* we gave it a bunch of researchers and we took a rapid feedback from them and improve the product before it became mature so the core design of PyTorch is very very researcher friendly
*  PyTorch is designed with users and just their feedback in mind
* PyTorch especially in its latest version sort of does also add features that make it easier to deploy models to production
* We built PyTorch event geared for production is you do research but when you want it to be production ready you just add functional annotations to your model which are like these one-liners that are top of a function

#### Hybrid Frontend
* We called a new programming model hybrid front-end because you can make parts of a model like compiled parts of my model and gives you the best of both worlds

#### Cutting-edge Applications in PyTorch
* one paper written by one person Andy Brock it was called smash where one neural network would generate the weights that would be powered
* hierarchical story generation so you would see a story with like hey I want a story of a boy swimming in a pond and then it would actually like generate a story that's interesting with that plot
* openly available github repositories, it's also just like very readable of work where you look at something you can clearly see like here are the inputs here is what's happening as far as it being transformed and here are the desired outputs

#### User Needs and Adding Features
* what users are wanting especially with being able to put models to production
* when they're exploring new ideas they don't want to be seeing like a 10x drop in performance
* online courses they want more interactive tutorials like based on a Python notebooks 
* some widgets they want first-class integration with collab

#### PyTorch and the Facebook Product
* I sort of think of it as being a separate entity from from Facebook which i think you know it definitely has its own life and community
* we also have a huge set of needs for products at Facebook whether it's our camera enhancements or whether it is our machine translation or whether it's our accessibility interfaces or our integrity filtering

#### The Future of PyTorch
* the next thing I was thinking was deep learning itself is becoming a very pervasive and essential confident in many other fields

#### Learning More in AI
* Ethos that that as students are yet trying to get into the field of deep learning either to apply it to their own stuff or just to learn the concepts it's very important to make sure you do it from day one
* my only advice to people is to make sure you do lesser but like do it hands-on

## Lesson 4

### Lectures

#### Single layer neural networks
* tensor
The primary data structure in TensorFlow programs. Tensors are N-dimensional (where N could be very large) data structures, most commonly scalars, vectors, or matrices. The elements of a Tensor can hold integer, floating-point, or string values.
<p align="center">
  <img src="./images/lesson-4/tensor.PNG" width="50%">
</p>

* [torch.sum()](https://pytorch.org/docs/stable/torch.html#torch.sum)
* [torch.mm()](https://pytorch.org/docs/stable/torch.html#torch.mm)
* [torch.matmul()](https://pytorch.org/docs/stable/torch.html#torch.matmul)
* [tensor.reshape()](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.reshape)
* [tensor.resize()](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.resize_)
* [tensor.view()](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view)

#### Networks Using Matrix Multiplication
* hyperparameter
The "knobs" that you tweak during successive runs of training a model. For example, learning rate is a hyperparameter.

#### Neural Networks in PyTorch
* neural network
A model that, taking inspiration from the brain, is composed of layers (at least one of which is hidden) consisting of simple connected units or neurons followed by nonlinearities.

* MNIST
A public-domain data set compiled by LeCun, Cortes, and Burges containing 60,000 images, each image showing how a human manually wrote a particular digit from 0–9. Each image is stored as a 28x28 array of integers, where each integer is a grayscale value between 0 and 255, inclusive.
<p align="center">
  <img src="https://github.com/agungsantoso/deep-learning-v2-pytorch/blob/master/intro-to-pytorch/assets/mnist.png?raw=1" width="50%">
</p>

* activation function
A function (for example, ReLU or sigmoid) that takes in the weighted sum of all of the inputs from the previous layer and then generates and passes an output value (typically nonlinear) to the next layer.
 
 * [nn.Sequential](https://pytorch.org/docs/master/nn.html#torch.nn.Sequential)

#### Network Architectures in PyTorch
* backpropagation
The primary algorithm for performing gradient descent on neural networks. First, the output values of each node are calculated (and cached) in a forward pass. Then, the partial derivative of the error with respect to each parameter is calculated in a backward pass through the graph.

* batch
The set of examples used in one iteration (that is, one gradient update) of model training.

* batch size
The number of examples in a batch. For example, the batch size of SGD is 1, while the batch size of a mini-batch is usually between 10 and 1000. Batch size is usually fixed during training and inference;

* cross-entropy
A generalization of Log Loss to multi-class classification problems. Cross-entropy quantifies the difference between two probability distributions

* epoch
A full training pass over the entire data set such that each example has been seen once. Thus, an epoch represents N/batch size training iterations, where N is the total number of examples.

* hidden layer
A synthetic layer in a neural network between the input layer (that is, the features) and the output layer (the prediction). Hidden layers typically contain an activation function (such as ReLU) for training. A deep neural network contains more than one hidden layer.

* logits
The vector of raw (non-normalized) predictions that a classification model generates, which is ordinarily then passed to a normalization function. If the model is solving a multi-class classification problem, logits typically become an input to the softmax function. The softmax function then generates a vector of (normalized) probabilities with one value for each possible class.

* optimizer
A specific implementation of the gradient descent algorithm.

* step
A forward and backward evaluation of one batch.
step size
Synonym for learning rate.

* stochastic gradient descent (SGD)
A gradient descent algorithm in which the batch size is one. In other words, SGD relies on a single example chosen uniformly at random from a data set to calculate an estimate of the gradient at each step.

* [nn.CrossEntropyLoss](https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss)
* [nn.LogSoftmax](https://pytorch.org/docs/stable/nn.html#torch.nn.LogSoftmax)
* [nn.NLLLoss](https://pytorch.org/docs/stable/nn.html#torch.nn.NLLLoss)
* [Optim Package](https://pytorch.org/docs/stable/optim.html)

#### Classifying Fashion-MNIST

#### Inference and Validation
#### Saving and Loading Models
#### Loading Image Data
#### Transfer Learning
#### Tips, Tricks, and Other Notes

### Quizes

### Notebooks
* [Tensors in PyTorch](https://github.com/agungsantoso/deep-learning-v2-pytorch/blob/master/intro-to-pytorch/Part%201%20-%20Tensors%20in%20PyTorch%20(Exercises).ipynb)
* [Neural networks with PyTorch](https://github.com/agungsantoso/deep-learning-v2-pytorch/blob/master/intro-to-pytorch/Part%202%20-%20Neural%20Networks%20in%20PyTorch%20(Exercises).ipynb)
* [Training Neural Networks](https://github.com/agungsantoso/deep-learning-v2-pytorch/blob/master/intro-to-pytorch/Part%203%20-%20Training%20Neural%20Networks%20(Exercises).ipynb)

## Lesson 5


## Lesson 6


## Lesson 7


## Lesson 8


## Lesson 9

## Challenge Project

## Credits
1. Images taken from lectures videos at [Intro to Deep Learning with PyTorch](https://www.udacity.com/course/deep-learning-pytorch--ud188)
2. [Machine Learning Glossary](https://developers.google.com/machine-learning/glossary/)