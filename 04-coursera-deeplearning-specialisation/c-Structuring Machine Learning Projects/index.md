# Deep Learning Specilisation - Course 3 / 5 
# Structuring Machine Learning Projects

## Week 1 
### Introduction to ML Strategy
#### 01 - Why ML Strategy ?

- after some time your model has 90% accuracy
- you have lot of ideas to improve:
  - collect more data
  - collect more diverse training set
  - traing algorithm longer
  - try different optimization (adam)
  - try bigger / smaller network
  - Try dropout/ L2 regularization
  - other network architecture
- figure out which ideas will help the most
- ml strategy is changing for deep learning (related to ml strategies)

#### 02 - Orthogonalization

- what to tune to achieve one effect: **orthogonalization**
- 1 node for tuning each "feature" seperately and **independent** 
- with orthogonal controls tuning is much easier
- for ml you have to do:
  - fit **training** set well on cost function $\approx$ human level performance
  - fit **dev** set 
  - fit **test** set 
  - performs well in real world
- **difficult**: early-stopping, because its a node affecting training and dev set performance


### Setting up your goal
#### 03 - Single number evaluiation metric

- one way is to watch precision and recall
- Precision: examples recognized as cats / percentage of actual cats
- Recall: of all real cat images , percentage of actual cats are correctly recognized
- often a tradeoff between Precision & Recall
- which classifier is better? (higher precision, lower recall or ...?)
- difficult to know, if we have 2 evaluation metrics
- $F_1$ Score = "Average" of P and R
$$ 
"Harmonic  mean": F_1 = \frac{2}{\frac{1}{P} + \frac{1}{R}}
$$

- Dev Set + Single real number evaluation metric
- speed up iterations (of improving model)
- imagine classifier with different score from distribution georgraphies
- compute average 

#### 04 - Satisficing and optimizing metric

- imagine you care about accuracy, running time
- cost = accuracy - 0.5 x running time ??
- maximize accuracy, subject to running time $\leq$ 100ms
- here accuracy needs optimizing, and running time is satisficing
- For N metrics: 1 optimizing, N-1 satisficing
- eg. wakewords / trigger words (like alexa, ok.google, hey siri, etc.)
  - accuracy of trigger word detection
  - number of false positive (randomly wake up without triggerwords)
  - combined: maximize accuracy, satisficing (s.t.) $\leq$ 1 false positive every 24h

#### 05 - Train, dev, test distributions

- dev set: development set or hold out cross validation set
- regions: us,uk, eur, SA, india, china, asia
- dont split set by regions, dont use different distribution
- take all of the data and randomly shuffle into dev/test
- other example: optimizing on dev set on loan approvals for medium income zip codes 
  - team suddenly decided to test it on low income zip codes
- choose a dev and test set reflecting data you expect to get in the future and consider important to do well on (from same distribution)

#### 06 - Size of the dev and test sets

- old way of splitting data:
  - 70, 30 split: train, test set
  - 60,20,20 split: train, dev, test set
  - was reasonable for dataset sizes like 100-10k
- modern ml dataset sizes like 1mio
  - 98,1,1
- for some applications you dont need high confidence for test set, just use train + dev, ship and dont care how well its doing, unusual
  - sometimes ppl talking about train+test, but evaluating with train + dev set

#### 07 - When to change dev/test sets and metrics

- imagine 
  - algorithm A: 3% error, but includes pornographic images
  - algorithm B: 5% error, but no pornographic images
  - Metric + Dev: Prefer A
  - You & Users: Prefer B
- Error $\frac{1}{m} \sum_1^m I {y_{pred} \neq y}$, just counts misclassified examples
- treating images equally (pornographic & non pornographic)
- can be fixed by adding $w^{(i)} = 1$, if x is non-porn, 10 if its porn. also adjust $\frac{1}{m}$ to $\frac{1}{\sum w^{(i)}$
- seperate steps:
  - place target to evaluate metrics
  - how to do well on this metric
- if doing well on your metric + dev/test set does not correspond to doing well on your applicationn, change your metric and/or dev/test set


### Comparing to human-level performance
#### 08 - Why human-level performance

- two main reason
- advance of deep learning, more feasable to become competitive with humans
- workflow of designing is much more efficient, if its sth that can be done by humans, too
- strong performance gain in the beginning, but slope goes down, mostly surpassing human-level performance
- approaches theoretical limit: Bayes optimal error (best possible error)
  - audio -> transcript, so noisy its impossible
  - image -> cat
  - not always 100%
  - best theoretical function for mapping
  - often near human level performance
- as long as your ml model is worse than humans, you can:
  - get labeled data from humans -> more data
  - gain insights from manual error analysis
  - better analysis of bias / variance

#### 09 - Avoidable bias
example cat classification
- human error 1%
- train error 8%, dev error 10%
- huge gap: focus on reducing bias -> bigger nn, gd longer

imagine:
- human error 7.5%
- train error 8 % , dev error 10%
- now we should focus on reducing variance -> regularization , ..

- human level error as a proxy / estimate for Bayes error
- for cv this is a good value
- call difference between bayes error and training error: avoidable bias
- call difference between training-dev error: variance
- some bias you cannot avoid

#### 10 - Understanding human-level performance

human level error as a proxy for bayes error
medical image classification
- (a) typical human - 3% error
- (b) typical doctor - 1% error
- (c) experienced doctor 0.7% error
- (d) team of xp doctor 0.5% error
- Bayes error $leq$ 0.5%, use d.
- for paper purpose you could also use (b) typical doctor, good enough to deploy application

error analysis example
- human (proxy for bayes): 1% or 0.5%
- train error 5%, dev error 6%
- avoidable bias: 4-4.5 %, variance 1%
- focus on bias reduction

2nd example
- train 1% error, dev 5%
- avoidable bias: 0-0.5%, variance 4%
- focus on variance reduction

3rd example
- train 0.7, dev 0.8%
- avoidable bias 0.2%, 0.1% variance


#### 11 - Surpassing human-level performance

- imagine: humans 0.5 - 1%
- training 0.3 %, dev error 0.4
- avoidable bias: ?? , variance : 0.1%
- overfitting by 0.2 ? reached bayes ? you dont know!
- if error is better than team of humans harder to rely on human intuition to improve itself
- options to make progress are less clear
- you might still be able to make progress 
- problems where ml significantly suprasses human-level performance
  - online advertising
  - product recommendations (movies, books)
  - logistics (predicting transit time)
  - loan approvals
  - all 4 learn from structured data 
- all are not natural perception problems
- humans are good at natural perception problems
- lots of data :) for pattern finding
- speech recognition can also surpass
- some image recognition
- medical task: ECG, skin cancer

#### 12 - Improving your model performance

- two fundamental assumptions
- fit training set pretty well (achieve low avoidable bias)
- training set performance generalizes pretty well to dev/test set (variance is low)
- summary to improve performance:
  - see avoidable bias: train error-human level
  - see variance: train - dev error
- reduce bias:
  - train bigger model
  - train longer, better optimization (momentum, RMSprop, Adam)
  - NN architecture, hyperparameter search (RNN, CNN)
- reduce variance:
  - more data
  - regularization (L2, dropout, data augmentation)
  - NN architecture, hyperparameter search

## Week 2
### Error Analysis
#### 13 - Carrying out error analysis

- imagine 90% accuracy, 10% error
- proposal: improve specifically for error with systems
- should you try make your cat classifier better at dogs ?

error analyis:
- get 100 mislabeled dev set examples
- count up how many are dogs
  - 5/100: dogs -> error just will be reduced to 9.5% 
  - 50/100: dogs -> error would be reduced to 5% (half error could be worth it)
- can evaluate one or multiple ideas
- create spreadsheets with ideas as columns, then count percentage
- estimation of worthwhile ideas

#### 14 - Cleaning up incorrectly labeled data

- incorrecly label (coming from data)
- mislabeleld label (coming from model)
- what to do? in training set:
  - robust to random errors in training set, if errors are reasonable random
  - systematic errors are a problem (all blacks: cats, white: not cat)
- what to do? in dev set? 
  - add incorreclty labeled to error analysis
  - depending on your other cases, e.g. 0.6 of 2% are because incorrect labels -> would be worth it 
  - dev set wants to help you between two classifiers A (2.1%) B (1.9%) 
- apply same process on dev / test set to make sure they come from the same distribution
- examine examples your algorithm got **right** as well it got **wrong**
  - consider but would take a long time
- train and dev/test data may now come from **slightly different** distributions

#### 15 - Build your first system quickly, then iterate

- advice: build quickly, then iterate
- speech recognition example
  - noisy background (cafe, car)
  - accented speech
  - far from mic
  - young children speech
  - stuttering
  - ...
- difficult to pick one, without spending too much time
- ml receipt:
  - setup dev/test set and metric
  - build initial system quickly (some learned system allows to prioritize)
  - use bias / variance analysis & Error analysis to prioritize next steps
- applies to all disciplines

