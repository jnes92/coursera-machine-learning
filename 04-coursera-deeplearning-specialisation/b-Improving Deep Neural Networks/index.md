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

## Week 2 
### Optimization algorithms
#### 15 - Mini batch gradient descent

- because iterative process
- quickly prototyping models is important
- large data -> slow
- fast optimization -> more efficiency


batch vs mini batch gd:
- vectorization allows to efficiently compute m examples
- what if m = 5.000.000 ?
- train on entire training set for 1 Step of Gradient descent
- faster algorithm, if you split training set to smaller training sets (mini-batches)
- e.g. mini-batch has 1000 examples each
- $X = [ X^{\{1\}}, X^{\{2\}}, ..., X^{\{5000\}} ]$ with ($n_x, 1000$)
- split Y accordingly with ($1, 1000$)
- mini-batch t: $X^{\{t\}}, Y^{\{t\}}$
  - () braces: training example
  - [] braces: layer in nn
  - {} braces: mini batch set
- **batch gd:** process complete training set in one batch at the same time
- **mini-batch gd**: algorithm process single mini batch at the same time 

pseudo-code for one epoch: 
> **epoch**: 1 single pass through training set
```python
for t = 1 ... 5000
#     gradient_descent( x{t}, y{t} )
    forward_prop on x{t} 
    # zI = wI x{t} + bI
    # Ai = gI x ZI
    # [...]
    # AL = gL + ZL
    J{t} = compute_cost() * 1/1000 + regularization_term
    backward_prop() on J{t}
    update_weights()
    # 
```

#### 16 - Understanding mini-batch gd

- batch gradient descent should decrease on every iteration
- mini-batch gradient descent **dont** decrease on **every** iteration
- downwards trend, but noisier with mini-batch

choosing mini-batch size
- if mini batch size = m $\rArr$ Batch gradient descent
- if mini batch size = 1 $\rArr$ stochastic gradient descent (every example is own mini-batch)
- for optimizing the cost function this means
  - batch gradient descent: low noise, large steps to the minimum
  - sgd: extremely noisy, but good direction most times, oscillate around minimum (wont stay there)
- in practice: sth in between 1,m
  - **BGD**: huge training set -> too long per iteration (for huge training set)
  - **SGD**: fast progress, noise can be removed by smaller learning rate, **loose** speed up from **vectorization**
  - in-between: fastest learning, because we have vectorization & make progress without waiting for entire training set
  - dont always hit local minimum exactly, but most times
- guidance
  - small training set (m<2000): BGD
  - typical min-batch size: 64, 128, 256, 512 (code runs faster if power of 2 some times), rarely 1024
  - make sure mini batch fits in CPU / GPU memory

#### 17 - Exponentially weighted averages

- faster algorithm than gradient descent
- needed to use exponentially weighted averages
- $v_0 = 0$
- $v_1 = 0.9 v_0 + 0.1 \Theta_1$
- $v_t = 0.9 v_{t-1} + 0.1 \Theta_t$
- plot moving average v(t)
- general: $v_t = \beta v_{t-1} + (1- \beta) \Theta_t$, with $\beta = 0.9$
- vt approximately averaging over $\approx \frac{1}{1- \beta}$ days, 
  - eg. $\beta = 0.9$ last 10 days temperature
  - e.g. $\beta = 0.98$ last 50 days
- averageing over larger window adapts slowly, more latency
  - $\beta = 0.5 \approx$ 2 days
  - averaging over 2 days, much more noisy, but adapts quickly to changes
- is called ***exponentially weighted moving averages***
- sth. in between works best :)

#### 18 - Understanding exponentially weighted averages

- $v_t = \beta v_{t-1} + (1- \beta) \Theta_t$
- how does it compute averages? 
  - $v_{100} = 0.1 \Theta_{100} + 0.9 ( 0.1 \Theta_{99} + 0.9 ( 0.1 \Theta_{98} + 0.9 v_{97} ) )$
  - $v_{100} = 0.1 \Theta_{100} + 0.1 x 0.9 \Theta_{99} + 0.1 (0.9)^2 \Theta_{98} + 0.1 (0.9)^3 \Theta_{97} + 0.1 (0.9)^4 \Theta_{96} + ...$
- is a weighted sum of exponentially decaying function .* the temperature values
- all coefficients sum up to $\approx 1$
- $1- \epsilon^{1 / \epsilon} = \frac{1}{e}$
- with $\epsilon = 1- \beta$

implementing exp. weighted averages:
```
V_theta = 0 
for i...m:
get next Theta_t
    V_theta = beta * v_theta + (1 - beta) Theta_t
```

#### 19 - Bias correction in exponentially weighted averages

- bias correction can make computation more accurate
- initialized $v_0 =0$ starts on y=0
- $v_1 = 0.98 v_0 + 0.02 \Theta_1$ -> $v_0$ will be ignored
- modify to make it more accurate for initial phase
- $\frac{v_t}{1- \beta^t}$
  - for t=2: $1-\beta^t = 1 - 0.98^2 = 0.0396 = \frac{0.0196 * \Theta_1 + 0.02 \Theta_2 }{0.0396}$
- for warming up averages
- bias correction is not needed if you dont care about initial phase, where averages are "warming up"

#### 20 - Gradient descent with momentum

- almost always faster than standard gd
- compute exponential moving average to update gradients
- gradient descent can take a lot of steps and slowly oscillate to the minimum, small learning rate needed
- vertically you want slower learning (too minimize oscillation), but horizontally you want faster learning
- momentum: on iteration t
  - compute dW, dB on current mini-batch
  - compute $v_{dw} = \beta v_{dw} + (1- \beta) dW$
  - $v_{db} = \beta v_{db} + (1- \beta) db$
  - $w = w- \alpha v_{dw}$, $b= b - \alpha v_{db}$
- momentum will be close to 0 for vertical line and just go for the horizontal direction
- now we have 2 hyperparameters $\alpha + \beta$
- $\beta = 0.9$ is often a good fit (average of last 10 gradients)
- alternative without $1-\beta$ :
  - $v_{dw} = \beta v_{dw} + dW$ 
  - works also fine, but less intuitive, because it affects scaling..

#### 21 - RMSprop

- Root mean square prop
- can also speed up gradient descent
- vertical axis: b, horizontal is parameter w
- On iteration t:
  - compute derivatives dW,dB on current mini-batch
  - $S_{dw} = \beta S_{dw}+ (1-\beta) dw^2$
  - element-wise squaring
  - $S_{db} = \beta S_{db}+ (1-\beta) db^2$
  - updating w,b:
  - $w = w - \alpha \frac{dw}{\sqrt{S_{dw}}}$
  - $b = b - \alpha \frac{db}{\sqrt{S_{db}}}$
- intuition: 
  - dW should be small -> update w faster
  - dB large -> small updatexy
  - will also flatten the updates vertically
  - larger learning rate possible
- combine RMSprop + momentum, parameter called $\beta_2$
- RMSprop was first made public in a coursera course :D

#### 22 - Adam optimization algorithm

- many researcher found many optimization algorithm, but they did not generalize good for all neural networks
- adam optimization is one of a few, who stands out
- basically: momentum + RMSprop

$$
v_{dw} = 0, s_{dw} = 0, v_{db} = 0, s_db = 0
$$
on iteartion t:
- compute dw,db using current mini-batch
- momentum update  with $\beta_1$
  - $v_dw = \beta_1 v_{dw} + (1- \beta_1) dw$
  - $v_db = \beta_1 v_{db} + (1- \beta_1) db$
- do RMS prop update  with $\beta_2$
  - $S_{dw} = \beta_2 S_{dw} + (1- \beta_2) dw^2$
  - $S_{db} = \beta_2 S_{db} + (1- \beta_2) db^2$
- correction  for v & S
  - $v_{dw}^{correction} = \frac{v_{dw}}{(1- \beta_1^t)}$
  - $v_{db}^{correction} = \frac{v_{db}}{(1- \beta_1^t)}$
  - $S_{dw}^{correction} = \frac{S_{dw}}{(1- \beta_2^t)}$
  - $S_{db}^{correction} = \frac{S_{db}}{(1- \beta_2^t)}$
- perform updates:
  - $w = w - \alpha \frac{v_{dw}^{corrected}}{\sqrt{S_{dw}^{corrected} + \epsilon}}$
  - $b = b - \alpha \frac{v_{db}^{corrected}}{\sqrt{S_{db}^{corrected} + \epsilon}}$
- hyperparamters for this algorithm
  - $\alpha$ needs to be tuned
  - $\beta_1$ common 0.9 (dw) for momentum
  - $\beta_2$ recommends: 0.999 moving avg $dw^2$
  - $\epsilon$ recommends $10^{-8}$
  - recommendations by adam paper team.
> **ADAM**: Adaptive moment estimation

#### 23 - Learning rate decay

- slowly reduce learning rate overtime
- algorithm can not converge 
- by reducing $\alpha$ we come closer to the minimum, by reaching convergence we can take smaller steps
- 1 epoch: 1 pass throug dataset (1 pass through all mini-batches)
$$
\alpha = \frac{1}{1+ decay\_rate * epoch\_num}
$$

Epoch | $\alpha$
- | - 
1 | 0.1
2 | 0.67
3 | 0.5 
4 | 0.4

other learning rate decay methods
- $\alpha = 0.95 ^{epoch_num} \alpha_0$
- $\alpha = \frac{k}{\sqrt{epoch_num}} \alpha_0$ or $\frac{k}{\sqrt{t}} \alpha_0$ 
- discrete staircase
- manual decay: manually controlling alpha by hand

#### 24 - The problem of local optima

- local optima: function can have multiple local optima, so algorithm can get stucked in one of them
- most point of gradient = 0 are saddle points, not local minima
- because we have like 20000 dimensions we will have all 20000 dimensions to be curved like quadratic (having a local optima) at this point
- chances are quite loooow $2^{-20000}$
- often its mixed, and we have a saddle point
- lot of intuition from 2D or 3D not transfer to high-dimensional problems
- problem of plateaus:
  - region where derivatives are close to 0 for a long time
- unlikely to get stuck in a bad local optima
- plateaus can make learning slow
  - momentum, RSMprop, adam can help here

## Week 3 - total 5h
### Hyperparameter tuning
#### 25 - Tuning process

- lot of hyperparameters needed
- tips for systematically organize tuning process
- learning rate $\alpha$, momentum $\beta$, adam ($\beta_1, \beta_2, \epsilon$), # layers, #hidden units, learninr rate decay, mini-batch size
- most important: $\alpha$, 2nd: $\beta$, hidden units, mini-batch, 3rd: layers, learning rate decay
- before: grid of 5x5 parameters for small parameter spaces
- recommend: choose random points
- reason: not evaluate 5 models, instead with random you really get 25 different models
- coarse to fine "scheme": mark the best fitting parameters and zoom in to this area and repeat process

#### 26 - Using an appropriate scale 

- $n^{[l]} = 50, ..., 100$ 
- #layers $L: 2 - 4$ 
- but for $\alpha  = 0.00001, ... , 1$ its hard to sample, because 90% is used between 0.1 - 1
- logarithmic scale (0.0001, 0.001, 0.001, 0.1, 1) you get more ressources for the small steps
- $r= -4 * np.random.rand()$ $\leftarrow r \in [-4,0]$
- $\alpha =  10^r$
- get low value to get $a= log_{10} 0.0001 = -4$, $b = log (1) = 0$ 
- input to $r: [a,b]$
- for exponentially weighted averages $\beta = 0.9, ... , 0.999$ (avg. of 10, 1000 values)
- $1- \beta = 10^r \rightarrow \beta = 1-10^r$ with $r \in [-3, -1]$

#### 27 - Hyperparameters tuning in practice: Pandas vs Caviar

- intuitions often dont transfer between domains (nlp, vision, speech, ads, logistics, ...)
- re evaluate occasionally
- 2 major ways:
  - babysitting one model (not enough computing power): adapt learning rate or parameters each day and decide if we can continue to grow lr
  - training models in parallel: try a lot of hyperparameters and pick the best afterwards
- also called:
  - panda approach (just 1-2 child with all the attention)
  - caviar: fish just lay a lot of eggs, hopefully one of them will do fine


### Batch Normalization
#### 28 - Normalizing activations in a network

- normalizing input can speed up learning, compute means, variance and normalize dataset
- in a deeper model, we have $X, a^{[1]}, a^{[2]}, ...$
- question: Can we normalize values of $a^{[2]}$ so as to train $W^{[3]}, b^{[3]}$ faster
- actually normalize $Z^{[2]}$ (some discussion about to normalize before or after activation function)
- given some intermediate values in NN $z^{[l](i)}$
- specific to one layer l, but left out to easier read it.

$$
\mu = \frac{1}{m} \sum_i z^{(i)}
$$
$$
\sigma^2 = \frac{1}{m} \sum_i (z_i - \mu)^2
$$
$$
z_{norm}^{(i)} = \frac{z^{(i)}- \mu}{ \sqrt{\sigma^2 + \epsilon} }
$$
$$
\tilde{z}^{(i)} = \gamma z_{norm}^{(i)}   + \beta 
$$
- $\gamma, \beta$ learnable parameters of model, 
- IF: $\gamma = \sqrt{\sigma^2 + \epsilon}, \beta = \mu$ it will $z_{norm}^{(i)} = \tilde{z}^{(i)}$
- use $\tilde{z}^{(i)}$ instead of $z^{(i)}$
- you dont want values to have mean = 0, variance like input features $X$

#### 29 - Fitting Batch Norm into a NN

- each hidden unit computes z + activation function to compute a
- $x \rightarrow^{w,b} z^{[i]} \longrightarrow_{BatchNorm(BN)}^{\beta, \gamma} \tilde{z}^{(i)} \rightarrow a^{[i]} = g^{[i]} (\tilde{z}^{(i)} )$ and repeat
- Parameters from network: $W,b$ + $\beta, \gamma$ for each Layer (not momentum $\beta$) 
- use optimization algorithm for all of them
- $\beta^{[l]} = \beta^{[l]} - \alpha d\beta^{[l]}$
- batch normalize in frameworks is only 1 line: `tf.nn.batch_normalization`
- batch norm is usually applied while working with mini-batches
- works the same way, like you would expect
- 1 detail for params: $W, \sout{b},\beta, \gamma$ during normalization b will be added and substracted (mean substraction step)
- dimensions of $\beta, \gamma: (n^{[l]},1)$ (like b before)

```
for t = 1 ... numMiniBatches
    compute forward_prop on X{t}
        in each hidden layer use BN to rpleace zL with zTildeL
    Use backprop to compute dw, (db), dbeta, dgamma of l
    update parameters w,beta, gamma (with gradient descent, + momentum, rmsprop, adam)
```
#### 30 - Why does Batch norm work? 

intuition:
- makes weights later more robust to changes for earlier layers
- imagine training with only black cats
- classifier might not do very well for colored cats
- data distribution changing "covariant shift"
- for a hidden layer inside all the inputs change every time (because $a^{[1]}, a^{[2]}$ change depending on the input)
- mean and variance will remain the same ($\beta^{[2]}, \gamma^{[2]}$)
- limits the amount of different values from previous layer, become more stable

slight regularization effect:
- each mini batch is caled by mean/variance computed on just that mini-batch
- there is some noise in it (because mini batch)
- similiar to dropout it adds some noise to each hidden layer activations
- combine with other regularization
- only works for small mini-batches (noise reduced for big mini-batch size)

#### 31 - Batch Norm at test time

- for test time: $\mu, \sigma^2$ estimate using exponentially weighted average (across mini-btaches)
- $X{1} -> \mu^{\{1\}[l]}$, ... -> $\mu$ and analog for $\sigma^2$
- $z_{norm} = \frac{z- \mu}{\sqrt{\sigma^2+\epsilon}}$
- $\tilde{z} = \gamma z_{norm} + \beta$
- also called "running average"

### Multi-class classification
#### 32 - Softmax Regression

- so far we used binary classification (0,1 / cat or non-cat)
- $C =$ #classes, here: 4
- $n^{[L]} = C$, here 4
- first node P(other | x), 2nd node: P(cat|x)
- $\hat{y} is (4,1)$ 
- Softmax Layer
  - Activation function
  - $t=e^{z^{[l]}}$
  - $a^{[l]} = \frac{e^{z^{[l]}}}{\sum_{j=i}^4 t_i} = \frac{t_i}{\sum_{j=i}^4 t_i}$
- summarize $a^{[l]} = g^{[l]} ( z^{[l]} )$
  - takes vector, output vectors
- model with softmax, but without hidden layer can seperate data into classes, with linear decision boundary

#### 33 - Training a softmax classifier

- name softmax comes from contrast to "hard max"
- hard max will just put one 1 to the biggest element, everything else get 0
- if c=2, softmax reduces to logistic regression, output layer will just have 2 numbers -> redundant, because they sum up to 1.
- Loss function
$$
L (\hat{y},y) = - \sum_{j=1}^C  y_j \log \hat{y}_j
$$
- if loss needs to be small, -log y hat 2 needs to be small -> make $\hat{y}_2$ big
$$
J (w,b,..) = \frac{1}{m} \sum_i^m L(\hat{y}, y)
$$

gradient descent with softmax:
- forward $z^{[l]} = a^{[l]} = \hat{y} \rightarrow L(\hat{y},y)$
- backprop $dz^{[l]} = \hat{y} - y$

### Introduction to programming frameworks
#### 34 - Deep learning frameworks

- learned algorithm from scratch to understand how they work
- as you start large models, its not practical to do so
- many good frameworks exist that can help you build
- frameworks:
  - caffe, caffe2
  - CNTK
  - DL4J
  - Keras
  - Lasgne
  - mxnet
  - PaddlePaddle
  - TensorFlow
  - Theano
  - Torch
- many frameworks evolve rapidly
- criteria for choosing dl frameworks
  - ease of programming (development, deployment)
  - running speed
  - truly open (open source + good governance)

#### 35 - Tensorflow

- motivating problem $J(w) = w^2 - 10w + 25 = (w-5)^2$, minimizing would be w=5
- pretend we dont know this :D 
- similiar structure can be used for neural networks

```python
import numpy as np
import tensorflow as tf

w = tf.Variable(0, dtype=tf.float32)    # Param we want to optimize

# get training data inside TF
# coefficients = np.array([[1.], [-10.], [25.]])
# x = tf.placeholder(tf.float32, [3,1]) # 3,1 array

# tf knows how to do derivatives, figures backprop out by itself
# cost = tf.add( tf.add(w**2, tf.multiply(-10., w)), 25))
cost = w**2 - 10*w + 25 # overlading +-*/ 
# cost = x[0][0]*w**2 - x[1][0]*w + x[2][0] # with training data X
train = tf. train.GradientDescentOptimizier(0.01).minimize(cost)

init = tf.global_variables_initializer()
session = tf.Session() 
session.run(init)
print (session.run(w))

session.run(train)
# session.run(train, feed_dict={x: coefficients})
print (session.run(w))

for i in range(1000):
    session.run(train)
print (session.run(w))
```

- only need to define the cost function
- better for cleanup while errors
```
with tf.Session() as session:
    session.run(init)
    print(session.run(w))
```
- tensorflow constructs a computation graph for the cost function
- tensorflow already built in backward propagation while constructing computation graph
- tensor flow documentation computation graph with operatioons $(.)^2$ instead of output $w^2$