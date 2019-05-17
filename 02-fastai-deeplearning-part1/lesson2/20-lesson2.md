# Lesson 2 : Data cleaning and production; SGD from scratch
## Times
17.5. video: 2h 


## Video

### Overview Text

First Part:
- starts with image classification with own data
  - image collection
  - parallel downloading
  - validation set
  - data cleaning

- getting image classifier ready for production -> online service

Second Part:
- train simple model from scratch, creating gradient descent loop 
- learn new terminology

### Intro
Forums Stuff:
- Checkout pinned lectures: Important Links
- Enable Notifications for threads with "watching"
- Use "Summarize this topics" to quickly scim through discussions with lot of replies

### Summary:
1. Computer Vision
2. NLP Applications
3. Tabular Data
4. Collaborative filtering - recommandation
5. Embeddings
6. Computer Vision II
7. NLP II 


approach of learning:
- watch videos like 3 times...
- whole game is like learning soccer as a kid (Perkins), start with the ball

### Notebook: Lesson 2 Download 
- teddy bear vs black bear vs grizzly 
- notebooks are often experimental -> you can skip cells, do whatever you like
- np.random.seed() with fixed number, will always make the dataset reproducable
- **you cant compare models with data from different validation sets!**

### Learning rates
- find biggest slope, for a good learning rate, with stable f'

### Cleaning Data
- plot top losses to decide which one are noisy images (mislabeled ?)
- every dataset in fastai contains properties x, y
- cleans mislabeled images inside data.valid (Validation set), replace with train afterwards
- use FileDeleter until you have some pages of valid data
- FileDeleter is an app inside JupyterNotebook (will this work in CoLab? )
- ipyWidgets
- noisy data is often not a huge problem
- problem is noisy data, if it is biased

### Putting model in production
- vast majority will run on cpu only
- gpu is good at much parallel tasks
- pretty unlikely to classify 60 images at the same time
- easier to do it on cpu, cheaper, more scalable
- trained model to predict called **inference**
- will use cpu if no gpu is available

- do this once on startup:
  - ImageDataBunch.single_from_classes( path for loading model)
  - other params need to be the same like on training
- `learn.predict (img)` for every request
- starlette is highly recommended
- python anywhere

## Problem solving

### Learning Rates
- **learning rate** too **high** -> **valid_loss** is really **high** 
- **learning rate** is too **low** -
  - **error_rate** gets better really **slow**
  - training loss > valid_loss (too low lr, too less epochs)

### Epochs
- just 1 Epoch: Train Loss >> valid_loss -> More Epochs
- too many Epochs: **overfitting**
  - is hard to overfit with deep learning
  - able to do so if you turn out all default options
  - train for really much epochs
  - error rate improves for lot of time and then decrease again
  
- overfitting is **NOT** happening if train_loss **<** valid_loss

## Going back a step: easy modelling ABC
- images are just pixel values
- matrix with 3rd dimension: **tensor**
- image classifier is a mathematical function to predict probability for classes
- argmax: find highest prob is and return me the index
- check source code of learn.predict?? to see it in action

how do we get to a function like this?
- starting with function: $y = ax + b = a_1 x + a_2 x_2$ and $x_2=1$
- $a_1, a_2$: coefficient, parameters
- if you remember dot product and matrix products this is easier written as: [see link](matrixmultiplication.xyz)
- $\vec{y} = X*\vec{a}$ 
- in matrix multiplication form we can do it in one PyTorch line

**FAQ:**
- metrics will always refer to the validation set
- lr 3e-3 is mostly a perfect start 
- after unfreezing `slice(xx from lr finder, 3e-3 / 10 = 3e-4`
- how much is enough data? 
  - if you found good learning rate, training for long time -> error is getting worse
  - but accuracy is still not good enough
  - -> Get more data
- what to do with unbalanced data classes? e.g. 10 bears + 1000 teddies
  - nothing.. :D 
  - paper for slightly better: take uncommon class & copy it -> oversampling
- resnet34 is only an architecture - not a pretrained model

## lets go to jupyter notebook **lesson2-sgd**
- SGD: Stochastic Gradient Descent
- create some random X and a 
- $X@a$ : matrix product (or vector matrix, or tensors or or)
- `x = torch.ones(n,2)`: 2 Params = rank 2, number of rows = n, number of columns = 2
- `x[:,0]` : stands for every single row, column 0 => uniform random number between (-1,1)
- `uniform_(-1,1)` pyhton function with underscore will replace returned value inline. 
- `plt` is referring to Matplotlib, mostly used in python (can do nearly everything, some other tools are better at some specific things)
- lets pretend we never knew the parameter for `a` 
- **if** we can find a way to find those 2 params to the 100 points we got
- we can also find a function to convert pixel values to class probabilty
- a neural network is literally doing the same thing
- pyTorch wants to find parameters (weights) to minimize error between points and the line `x@a` 
- loss: average of error from each point
- regression problem: variable dependent is continuous values (number)
- for regression the most common error function is **mse**: mean squarred error
- tensors should be float numbers so dont do `tensor(-1,1)` will be int. use `tensor(-1.,1)`
  - check `a.type()` to be FloatTensor

Now we need to do **optimization**!

- from calculus we know from the derivative which value will change the error

### Gradient Descent

- dont calculate it by hand use `.grad`
- `loss.backward()` can give us the derivative 
- substract gradient from a, because we wont to do the opposite to keep loss small
- lr: learning rate

GD algorithm:
- pick random value as starting point
- calculate gradient, tells which direction to go to reduce loss
- if gradient is very big you can jump over the local minima - won´t help
- jumping all the way to the minima
- multiplied by learning rate to dont jump too far.
- remind of lr: 
  - too high, loss will get higher each time
  - too low: much multiplication, but takes a lot of time
- matplotlib is able to animate diagrams, visualizing is quite easy
- visualize 15k parameters is hard to do, but its the same thing

### SGD Stochastic Gradient descent:
- dont calulate whole file at once, but use mini-batches
- grab some part of the data und update weights and loop it

## How to get the right function
- underfitting is like presentic quadratic function with a linear architecture
- overfitting is using a complex function to fit all points exactly
- just right is what we want 
  - not about the right count of parameters
  - use regularization also

## Glossar

- **inference**: running model for production
- **epoch**: one complete run through all of our datapoints, too many will overfit
- **learning rate**: multiply gradient by, to decide how much to update the wiehgts
- **loss function**: how close you are to correct answer
- **error rate**: error_rate?? or doc(error_rate) = 1 - accuracy
- **tensor**: sounds scary, is for physicians - means array of regular shape (rectangular or cubic shape), e.g. image (rows x colums x color), we dont say dimension -> say rank -> image is rank 3, rank1 tensor = vector, rank2 tensor = matrix -> just call everything tensor for computers :D
- **mini-batch**: random bunch of points to update weights
- **SGD** GD + mini-batching data
- **Architecture / Model**: mathematical function to fit parameters to
- **parameters** coefficients / weights


## funny things
- there are no math people, its all culture and expectation:
  - check out "There´s no such thing as "not a math person"
  - some cultures dont even know this sentence :D 

## Todo Lesson 2:
- [ ] Complete Glossar 
- [ ] put some models to production with starlette
- [ ] check plots with high lr, low lr for lesson2 - sgd

## Reminder
- check production starter kits available
- new interactive GUI in notebook for using model to find and fix mislabeld images
- identify objects from google maps
- fastai contribute? 
- trash detection with drones? 
- NLP for Code language detection ?!
- check *ipywidgets* for GUI Programming inside jupyter

## Ideas:
- evtl. Workshop with lesson2-download for building image classifiers? DNUG/ BM?
- Brownbag zu Basics for Machine Learning: How to Gradient descent with PyTorch (lesson2-sgd)