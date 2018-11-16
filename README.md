# PyTorch Scholarship Challenge 2018/2019 Notes


![GitHub](https://img.shields.io/github/license/mashape/apistatus.svg)



A collection of notes on PyTorch Scholarship Challenge 2018/2019.

Contributions are always welcome!

<!-- toc -->

- [AMA](#ama)
- [Lesson 2](#lesson-2)
  * [Classification Problems](#classification-problems)
  * [Decision Boundary](#decision-boundary)
  * [Perceptrons](#perceptrons)
  * [Why "Neural Networks"?](#why-neural-networks)
  * [Perceptrons as Logical Operators](#perceptrons-as-logical-operators)
  * [Perceptron Trick](#perceptron-trick)
  * [Perceptron Algorithm](#perceptron-algorithm)
  * [Non-Linear Regions](#non-linear-regions)
  * [Error Functions](#error-functions)
  * [Log-loss Error Function](#log-loss-error-function)
  * [Discrete vs Continous](#discrete-vs-continous)
  * [Softmax](#softmax)
  * [One-Hot Encoding](#one-hot-encoding)
  * [Maximum Likelihood](#maximum-likelihood)
  * [Cross-Entropy](#cross-entropy)
  * [Multi-Class Cross Entropy](#multi-class-cross-entropy)
  * [Logistic Regression](#logistic-regression)
  * [Gradient Descent](#gradient-descent)
  * [Feedforward](#feedforward)
  * [Backpropagation](#backpropagation)
  * [Overfitting & Underfitting](#overfitting--underfitting)
    + [Overfitting](#overfitting)
    + [Underfitting](#underfitting)
  * [Early Stopping](#early-stopping)
  * [Regularization](#regularization)
  * [Dropout](#dropout)
  * [Local Minima](#local-minima)
  * [Random Restart](#random-restart)
  * [Momentum](#momentum)
- [Lesson 3](#lesson-3)
- [Lesson 4](#lesson-4)
- [Lesson 5](#lesson-5)
- [Lesson 6](#lesson-6)
- [Lesson 7](#lesson-7)
- [Lesson 8](#lesson-8)
- [Lesson 9](#lesson-9)
- [Credits](#credits)

<!-- tocstop -->

;

## AMA

* [2018-11-09](https://drive.google.com/open?id=1Oa_NfQkqnrqcn29p7v5Hv9ttiHOnvXsc)
* [2018-11-13](https://drive.google.com/open?id=1SMxlFQzBi4yukHu-mwhXTz79MApUiiHS)
* [2018-11-14](https://drive.google.com/open?id=1SMxlFQzBi4yukHu-mwhXTz79MApUiiHS)

## Lesson 2

### Classification Problems
  The problem of identifying to which of a set of categories (sub-populations) a new observation belongs.

  <p align="center">
    <img src="./images/lesson-2/classification-problems.PNG" width="50%">
  </p>

### Decision Boundary
  The separator between classes learned by a model in a binary class or multi-class classification problems. For example, in the following image representing a binary classification problem, the decision boundary is the frontier between the blue class and the red class:

  * Linear Boundaries
    <p align="center">
      <img src="./images/lesson-2/linear-boundaries.PNG" width="50%">
    </p>

  * Higher Dimensions
    <p align="center">
      <img src="./images/lesson-2/higher-dimensions.PNG" width="50%">
    </p>

### Perceptrons
  A system (either hardware or software) that takes in one or more input values, runs a function on the weighted sum of the inputs, and computes a single output value. In machine learning, the function is typically nonlinear, such as ReLU, sigmoid, or tanh.

  In the following illustration, the perceptron takes n inputs, each of which is itself modified by a weight before entering the perceptron:

  <p align="center">
    <img src="./images/lesson-2/perceptrons.PNG" width="50%">
  </p>

  A perceptron that takes in n inputs, each multiplied by separate
  weights. The perceptron outputs a single value.

  Perceptrons are the (nodes) in deep neural networks. That is, a deep neural network consists of multiple connected perceptrons, plus a backpropagation algorithm to introduce feedback.

### Why "Neural Networks"?

  <p align="center">
    <img src="./images/lesson-2/why-neural-network.PNG" width="50%">
  </p>

### Perceptrons as Logical Operators

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

### Perceptron Trick
<p align="center">
  <img src="./images/lesson-2/perceptron-trick.PNG" width="50%">
</p>

### Perceptron Algorithm
<p align="center">
  <img src="./images/lesson-2/perceptron-algorithm.PNG" width="50%">
</p>

### Non-Linear Regions
<p align="center">
  <img src="./images/lesson-2/non-linear-regions.PNG" width="50%">
</p>

### Error Functions
<p align="center">
  <img src="./images/lesson-2/error-functions.PNG" width="50%">
</p>

### Log-loss Error Function
<p align="center">
  <img src="./images/lesson-2/log-loss-error-function.PNG" width="50%">
</p>

### Discrete vs Continous
<p align="center">
  <img src="./images/lesson-2/discrete-vs-continous.PNG">
</p>

### Softmax
A function that provides probabilities for each possible class in a multi-class classification model. The probabilities add up to exactly 1.0. For example, softmax might determine that the probability of a particular image being a duck at 0.67, a beaver at 0.33, and a walrus at 0. (Also called full softmax.)

<p align="center">
  <img src="./images/lesson-2/softmax.PNG" width="50%">
</p>

### One-Hot Encoding

A sparse vector in which:

* One element is set to 1.
* All other elements are set to 0.

One-hot encoding is commonly used to represent strings or identifiers that have a finite set of possible values. For example, suppose a given botany data set chronicles 15,000 different species, each denoted with a unique string identifier. As part of feature engineering, you'll probably encode those string identifiers as one-hot vectors in which the vector has a size of 15,000.

<p align="center">
  <img src="./images/lesson-2/one-hot-encoding.PNG" width="50%">
</p>


### Maximum Likelihood
<p align="center">
  <img src="./images/lesson-2/maximum-likelihood.PNG" width="50%">
</p>

### Cross-Entropy

A generalization of Log Loss to multi-class classification problems. Cross-entropy quantifies the difference between two probability distributions.

<p align="center">
  <img src="./images/lesson-2/cross-entropy.PNG" width="50%">
</p>

### Multi-Class Cross Entropy
<p align="center">
  <img src="./images/lesson-2/multi-class-cross-entropy.PNG" width="50%">
</p>

### Logistic Regression
A model that generates a probability for each possible discrete label value in classification problems by applying a sigmoid function to a linear prediction. Although logistic regression is often used in binary classification problems, it can also be used in multi-class classification problems (where it becomes called multi-class logistic regression or multinomial regression).
<p align="center">
  <img src="./images/lesson-2/logistic-regresssion.PNG" width="50%">
</p>

### Gradient Descent
A technique to minimize loss by computing the gradients of loss with respect to the model's parameters, conditioned on training data. Informally, gradient descent iteratively adjusts parameters, gradually finding the best combination of weights and bias to minimize loss.
<p align="center">
  <img src="./images/lesson-2/gradient-descent.PNG" width="50%">
</p>

### Feedforward
<p align="center">
  <img src="./images/lesson-2/feedforward.PNG" width="50%">
</p>

### Backpropagation
The primary algorithm for performing gradient descent on neural networks. First, the output values of each node are calculated (and cached) in a forward pass. Then, the partial derivative of the error with respect to each parameter is calculated in a backward pass through the graph.
<p align="center">
  <img src="./images/lesson-2/backpropagration.PNG" width="50%">
</p>

### Overfitting & Underfitting
#### Overfitting
Creating a model that matches the training data so closely that the model fails to make correct predictions on new data.

#### Underfitting
Producing a model with poor predictive ability because the model hasn't captured the complexity of the training data. Many problems can cause underfitting, including:

* Training on the wrong set of features.
* Training for too few epochs or at too low a learning rate.
* Training with too high a regularization rate.
* Providing too few hidden layers in a deep neural network.

<p align="center">
  <img src="./images/lesson-2/overfitting-and-underfitting.PNG" width="50%">
</p>

### Early Stopping
A method for regularization that involves ending model training before training loss finishes decreasing. In early stopping, you end model training when the loss on a validation data set starts to increase, that is, when generalization performance worsens.
<p align="center">
  <img src="./images/lesson-2/early-stopping.PNG" width="50%">
</p>

### Regularization
The penalty on a model's complexity. Regularization helps prevent overfitting. Different kinds of regularization include:

* L1 regularization
* L2 regularization
* dropout regularization
* early stopping (this is not a formal regularization method, but can effectively limit overfitting)

<p align="center">
  <img src="./images/lesson-2/regularization.PNG" width="50%">
</p>

### Dropout
A form of regularization useful in training neural networks. Dropout regularization works by removing a random selection of a fixed number of the units in a network layer for a single gradient step. The more units dropped out, the stronger the regularization. This is analogous to training the network to emulate an exponentially large ensemble of smaller networks.
<p align="center">
  <img src="./images/lesson-2/dropout.PNG" width="50%">
</p>

### Local Minima
<p align="center">
  <img src="./images/lesson-2/local-minima.PNG" width="50%">
</p>

### Random Restart
<p align="center">
  <img src="./images/lesson-2/random-restart.PNG" width="50%">
</p>

### Momentum
A sophisticated gradient descent algorithm in which a learning step depends not only on the derivative in the current step, but also on the derivatives of the step(s) that immediately preceded it. Momentum involves computing an exponentially weighted moving average of the gradients over time, analogous to momentum in physics. Momentum sometimes prevents learning from getting stuck in local minima.
<p align="center">
  <img src="./images/lesson-2/momentum.PNG" width="50%">
</p>

## Lesson 3


## Lesson 4


## Lesson 5


## Lesson 6


## Lesson 7


## Lesson 8


## Lesson 9

## Credits
1. Images taken from lectures videos at [Intro to Deep Learning with PyTorch](https://www.udacity.com/course/deep-learning-pytorch--ud188)
2. [Machine Learning Glossary](https://developers.google.com/machine-learning/glossary/)