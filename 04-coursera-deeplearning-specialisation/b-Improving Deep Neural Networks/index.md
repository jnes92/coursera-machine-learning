# Deep Learning Specilisation - Course 2 / 5 
# Improving Deep Neural Networks
# Hyperparameter tuning, Regularization and Optimization

## Week 1 - total 8h
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

- for high variance -> try regularization first
- prevent overfitting 

for **Logistic regression**:
- $J (w,b) = \frac{1}{m} \sum L(\hat{y},y) + \frac{\lambda}{2m} || w || ^2$
- regularization for b is not needed (just 1 number)
- $L_2$ regularization: $|| w ||_2^2  = \sum_{j=1}^{n_x} w_j^2 = w^Tw$
- $L_1$ regularization $\frac{\lambda}{m} \sum_{i=1}^{n_x} | w | = \frac{\lambda}{m} || w ||_1$
- $L_2$ is used more often
- $\lambda$ : Regularization parameter, hyperparameter (*lambd for python, because reserverd keyword*)

for **neural network**
- $J(w^{[1]}, b^{[1]} ..) = \frac{1}{m} \sum_{i=1}^m L(\hat{y},y) +  \frac{\lambda}{2m} \sum_{l=1}^L || w^{[l]} || ^2$ 
-  $|| w^{[l]} ||_F ^2 = \sum_{i=1} \sum_j ( w_{ij}^{[l]} )^2$
-  $w: (n^{[l]}, n^{[l-1]})$
-  called "Frobenius norm", instead of $L_2$

for gradient descent this updates 
- $dw =$ from backprop + $\frac{\lambda}{m} w^{[l]}$
- $w^{[l]} = w^{[l]} - \alpha dw^{[l]} = w^{[l]} \alpha[(from backprop) + \frac{\lambda}{m} w^{[l]}]$
- $w^{[l]} = w^{[l]} - \frac{\alpha \lambda}{m} w^{[l]} - \alpha$ (from backprop)
- also called "weight decay", because in the end you multiply with $w(1- \frac{\alpha \lambda}{m})$

#### 05 - Why regularization reduces overfitting

- $J(w^{[l]}, b^{[l]}) = \frac{1}{m} \sum_{i=1}^m L(\hat{y},y) +  \frac{\lambda}{2m} \sum_{l=1}^L || w^{[l]} ||_F ^2$ 
- intuition of zero-ing out units is wrong, they will be just very small
- intuition tanh:
  - if lambda is large parameters $w$ will be small, so will $z$
  - g(z) would be roughly linear
  - if every layer is linear, your network is just a linear function
- implemtation tip: J regularized = J default + regularization term
- plot Cost function, to iterations (also with second term)

#### 06 - Dropout Regularization

- 0.5 chance of keeping or removing each nodes / units
- remove all the in/ out lines
- train with reduced network
- smaller network for training reduces overfitting

**Inverted dropout**
- illustrate with layer $l=3$, $keep.prob = 0.8$
- d3 (dropout for layer 3)
```python
d3 = np.random.rand(a3.shape[0], a3.shape[1]) < keep.prob
a3 =  np.multiply(a3,d3) # a3 .* d3, dotwise
a3 /= keep.prob
```

- most common dropout technique: inverted dropout
  
making predictions at test time:
- $a^{[0]} = X$
- no dropout for test time?
  - $z^{[1]} = w^{[1]} a^{[0]} +b^{[1]}$ -> $\hat y$

#### 07 - Understanding Dropout

intuition: cant rely on any one feature, so have to spread out weights
- shrinking squared norm of the weights
- is formally an adaptive form of $L_2$
- keep.prob can vary between layers
  - for big layers you can dropout 0.5
  - for small layers 0.7 or even 1
  - dropout is even possible for input layers (not used often, close to 1)

implementation tips
- often used in computer vision
- Downside: J is not defined, so plotting or analyzing is much harder
- start with dropout 1, make sure its working and then activating dropout

#### 08 - Other regularization methods

- data augmentation
  - eg. with images you can: flip x, crop, rotate, zoom
  - eg. for digit: distortions,
- early stopping
  - while running gradient descent plot training error / or J 
  - plot dev set error in the same graph
  - early stopping: stop at Iteration XY (e.g. 50%)
  - downside: 
    - Optimize Cost function J (Gradient descent, Adam, ...)
    - Not overfit (Regularization, ...)
    - if seen as seperate task you will mix "Orthogonalization" with early stopping
    - cant work independently on the two tasks :-( 

### Setting up your optimization problem
#### 09 - Normalizing inputs

- two steps: 
  - substract mean $\mu = \frac{1}{m} \sum x^{(i)}$
  - $x = x - \mu$
  - normalize variance $\sigma^2 = \frac{1}{m} \sum_{i=1}^{(i)} .* ^2$
  - $x /= \sigma^2$
- why do we want to normalize? 
  - if features are on different scales ($x_1: 1..1000, x_2: 0..1$)
  - cost function will be more symmetric for normalized
  - we need a smaller learning rate for unnormalized

#### 10 - Vanishing / Exploding gradients

- $\hat{y} = w^{[l]} w^{[l-1]} w^{[l-2]} ... w^{[2]} w^{[1]}$
- $a^{[2]} = g(z^{[1]}) = g(w^{[2]} a^{[1]} )$
- e.g. $\hat y = 1.5^{L-1} x$ explodes really fast
- or if values smaller 1, decrease exponentially
- $w^{[l]} > I (or 1)$: Explode exponentially
- $w^{[l]} < I (or 1)$: Vanish exponentially

#### 11 - Weight Initialization for Deep Networks

- better choice of random initialization can improve the vanish / exploding problem
- for large n -> smaller weights $w$
- $Var(w) = \frac{2}{n}$
- `WL = np.random.randn(entershapehere) * np.sqrt(2/n[l-1])`

other variants:
- version is assuming ReLu activation
- **Xavier** for tanh: constant 1 instead of ()  `np.sqrt(1/n[l-1]) `
- `np.sqrt(2/n^[l-1]+n^[l])`
- could be also a hyperparameter

#### 12 - Numerical approximation of gradients

checking your derivative computation:
- $\frac{f(\Theta + \epsilon) - f(\Theta - \epsilon)}{2 \epsilon} \approx g(\Theta)$ 
- exactly definition of f' for epsilon limit against 0

#### 13 - Gradient Checking

- Take all Parameters W,b and reshape into big vector $\Theta$
  - $J(w,b,...) = J(\Theta)$
- Take all gradients dW, db and reshape into a big vecor $d\Theta$

grad(ient) check:
- for each i:
- $d\Theta_\approx [i]$ = J(... +e) - J(... -e) / 2e 
- $d\Theta_\approx  \approx d\Theta$ ???
- compute euclidian distance $\frac{ || d\Theta_\approx  -d\Theta ||}{||d\Theta_\approx||_2  + ||d\Theta||}$
- great: $10^{-7}$, ok: $10^{-5}$, bad: $10^{-3}$

#### 14 - Gradient Checking Implementation Notes

- dont use grad check in training, only for debugging
- if algorithm fails grad check, look at components to try to identify bug, compare values of 
  - e.g. db is incorrect, but dw looks good all the time, -> check db functions
- remember regularization term, if you used it
- does not work with dropout
- run with random initialization (can happen that its just correct if init with $\approx$ 0 for w,b)