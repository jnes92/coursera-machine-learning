# Deep Learning Specilisation - Course 2 / 5 
# Improving Deep Neural Networks
# Hyperparameter tuning, Regularization and Optimization

## Week 1
### Setting up your Machine Learning Application
#### 01 - Train / Dev / Test sets

- good choice for datasplit is important
- applied ml is highly iterative process
- choose layers, hidden units, learning rates, activation functions
- in practice: **repeat** idea $\rarr$ code $\rarr$ experiment
- different fields of ml: intuition dont transfer to other areas
- impossible to guess all the hyperparameters


train, dev, test set:
- typically you will split your data to 3 parts:
  - training set
  - hold-out cross validation set / development set
  - test set
- train model with training set
- evaluate with cv / dev set
- for final model evaluate on test set

previous era: 
- split 70/30 (train, test split) or maybe (60,20,20)
- was best practice a long time
- for data up to 10k its perfectly fine

big data:
- for data over 1 mio 
- better: maybe 10k for dev or test set, than 20%
- ratio is like: 98/1/1
- or even harder for more data: 99.5 / 0.4 / 0.1

mismatched train / test distribution:
- care if uploaded pictures are from different sources (upload from webpage, app)
- make sure dev and test set come from same distribution


> not having a test set might be okay (only cv/dev set), if you not completely need an unbiased model

#### 02 - Bias & Variance

- easy to learn, difficult to master
- bias variance tradeoff is often not talked about
- high bias: underfitting, not matching good
- high variance: overfitting, matching too good, not generalized
- just right is in between
- high dimensional problems: hard to visualize

<!-- TODO: Make it a table?  -->
cat classification
- train set error: 1 %
- cv/dev set error: 11%
- very well on training set, poor on development set
- high variance

other example result
- train set error: 15%
- dev set error: 16%
- in comparison to human $\approx 0$%
- underfitting -> high bias

more example:
- train 15%
- dev: 30%
- high bias &
- high variance

good example:
- train 0.5 %
- dev: 1 % 
- low bias, low variance

Optimal error (Bayes) error $\approx$ 0%

#### 03 - Basic Recipe for ML

- high bias ? (training data performance)
  - try bigger network
  - train longer
  - (NN architecture search)
- NO high bias $\rArr$  high variance ? (dev set performance)
  - More data
  - regularization
  - (NN architecture search)
- Done :)

- bias / variance trade off
  - pre-deep learning there were not so much tools that only reduces one of them
  - in modern dl: if you can keep getting more data you can reduce bias and variance without hurting the other..

### Regularizing your neural network
#### 04 - Regularization
#### 05 - Why regularization reduces overfitting
#### 06 - Dropout Regularization
#### 07 - Understanding Dropout
#### 08 - Other regularization methods

### Setting up your optimization problem
#### 09 - Normalizing inputs
#### 10 - Vanishing / Exploding gradients
#### 11 - Weight Initialization for Deep Networks
#### 12 - Numerical approximation of gradients
#### 13 - Gradient Checking
#### 14 - Gradient Checking Implementation Notes