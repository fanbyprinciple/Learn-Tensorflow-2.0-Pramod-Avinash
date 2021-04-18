# Learn-Tensorflow-2.0-Pramod-Avinash
Implementing code given in the book


# Chapter 1 

Introduction to Tensorflow 2.0

Eager execution
1. Tensorflow 2.0 does not require graph definition
2. Tensorflow 2.0 does not require session execution
3. Need not initialise variables
4. Doesn't require variable sharing via scopes

tf.function to create graph

tf.keras -  simplicity of keras

image default datasets

imdb_reviews, squad
mnist, imagenet2012, coco2014, cifar10
moving_mnist, starcraft_video, bair_robot_pushing_small, Nsynth, titanic, iris

![](img/chapter1.png)

# chapter 2

Supervised Learning and Tensorflow

## Linear Regression Model using TensorFlow and Keras

![](img/ch2_linear_regression.png)

Custom Linear Regression on Diabetes test

![](img/ch2_linear_regression_custom.png)

How to train the data though.

Page 51.

## Logistic Regression

![](ch2_logistic_regression_seaborn.png)

https://colab.research.google.com/drive/1KOOjBvur0UeB_MDClIX6CoWrqd0JnZVo#scrollTo=Oxcj9RalOuq2

page 57

## bagging and boosting

It is a technique wherin we build independent models/ predictors using random subsample/bootstrap of data of each of the models/predictors.

bagging all models train independently and finally the average is taken

boosting all models train sequentially and each model learns from the previous model.

## Gradient Boosting

Here instead of incrementing the weights of the misclassified learner we optimize the lossfunction of the previous learner.

![](ch2_gradient_boosting.png)

# chapter 3

Neural Networks and Deep Learning with TensorFlow

## Neural net fashion mnist

simple nn

![](ch3_simplenn.png)

Deep nn

![](ch3_deepnn.png)

## Estimator using iris

