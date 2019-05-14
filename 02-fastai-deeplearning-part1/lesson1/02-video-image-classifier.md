# Lesson 1 Image Classification

## Overview Text:
- key outcome is a trained image classfiier which recognizes pet breeds 
- use of transfer learning
- how to analyze the model to understand its failure
- model is doing mistakes in same areas that even breeding experts can make mistakes

- discuss overall approach of the course
- unusal being top-down rather than bottom-up
- starting with practical applications and dig deeper 
- learning theroy as needed
- approach takes more work for teachers to develop

- how to set most important hyper-parameter when training: learning rate
- using Leslie Smiths fantastical learning - rate - finder method
- labeling, features from fast ai to easily add labels to images

- maybe checkout PyTorch -> suggest after completing a few lessons

## Video 

### intro
- setup gpu should be ready
- notebook tutorial should be tested
- code is in python

- notebook is a interactive experiment
- 3 years of experience with this course
- pretty good to watch end-to-end
- take information and afterwards go throug more slowly
- trying things out, extend in own way
- dont try to understand everything the first time
- lessons get faster and more difficult

### about jeremy howard
- over 25 years ml
- consulting, self started startups
- kaggle president, competitor (nr 1)
- co founder of fast ai

### making deep learning accessible 
- create software
- do education
- reasearch to make it easier to use
- community help each others through forum

### overall 
- 7 lessons
- 8-10 hours per lesson
- different ways to go on

- you will be able to
  - classify pet photos
  - identify movie review sentiment
  - predict supermarket sales
  - recommendation system

claims about deep learning:
- black box
- needs too much data
- needs ml PhD
- only for vision
- needs lot of GPUs
- not really "AI"

approach is different from academics
- start with code
- learn how it works before why it works
- create lots of models
- study them

### Notebook lesson 1
- % called directives to jupyter notebook
- fastai library (fast.ai - company name)
- use fastai or PyTorch
  - one of most popular libraries
  - do lot more, quicker with PyTorch 
  - (Tensorflow 2.0 is released by now)
- fast ai has 4 applications
  - computer vision
  - natural language text
  - tabular data
  - collaboritve filtering

- import * is not good for standard productive code in python
- for matlab its nice to have everything included
- 2 ways: SoftwareEngineering & Scientific
- for Jupyter Notebooks : experimental
- for Production use SWE approach
- different rules for scientific are going to be used

datasets
- academic dataset & kaggle dataset
- strong baselines
- see how well you have gone in competition
- cite dataset
- before you had only Dog vs Cat (was difficult before, but is easy nowadays without tuning)
- deep learning can do harder problems today
- fine grained classification for pet breeds

- downloading data with untar_data(url)
- PathObj are better than strings platform independent path/'annotations'
- how do we get labels? all files are just in a flat folder
- but labels are inside filename
- ImageDataBunch is all you need to train a model
  - contains 2 or 3 datasets (training, validition maybe test data)
  - from_name_re (regular expression -> extracting the label with a pattern)
  - need to specify size, gpu has too apply same instructions to all images to be fast
- if images are different sizes it cant do this -> current short coming
- part 1 will do square images, part2 will show how to use rectangles
- size = 224 will generally just work
- normalization: all of your data should be about the same size
- ds_tfms = combination of cropping and resizing and some data augmentation

- remember to look at the dataset

### Training the model

- model gets learned by a learner
- learner is a general concept 
- ConvLearner will create a convolutional neural network
- lot of different variants to create some
- resnet works really well
- for getting started just should choose size 34 / 50 
- start with small one and check if it is good enough
- first time it will download some pre trained weights
- model has been trained before for some task
- 1.5 million pictures from a thousand different categories
- download those weights learned from imageNet
- this is called transfer learning
- 1/100 less training time, 1% of data
  
- overfitting: dont learn in general, just cheat on this particular data
- to check this we use a validation set
- metrics will be outprinted from validation set
- by creating databunch we already splitted it

- 2018 best way to learn is using fit_one_cycle 
- how to select epoch / cycle - later -> choose 4 by now
- error rate of 6% ... 94 % is correct from start
- compare our solution with academics paper (2012)

- focus on code: What comes in, what comes out!
- fastai (easy use libraries) is keras (runs on top of tensorflow)
- training time is half
- lines of code is 1/6 (e.g. 31 lines vs 5)
- new state of the art result in natural language processing built in fastai

where can it take you
- non profit deltaAnalytics: system with old mobile phones to listen for chainsaw noises -> deep learning to alert rangers -> Google Brain researcher
- Clara : neural net music generator -> open ai (top 3 research with deepmind)
- advice *pick one project, do it really well and make it fantastic*
- take domain knowledge and combine with deep learning
- splunk and tensorflow: algorithm for fraud detection
- hotdog or not app from fast ai student

### faq
- architecture: dawnbench benchmark resnet is fast and good enough
- for edge computing: run it on server or use special architecture

### RESULTS

- interprete model with passing learn object
- prints 3 things: interpreted as, is actually, loss prob 
- confusion matrix shows for every categories times of wrong
- for many classes use 'most_confused'

### Improve the model: Unfreezing, fine-tuning & learning rates

- network has many layers, which all compute
- added extra layers in the end and only train those ones
- error rate grows :( 
- resnet 34 has 34 layers
- layer 1 with coefficients as images (finding lines)
- layer 2 takes result from layer 1 -> learns corners or circles (finding shapes)
- layer 3 can find patterns of shapes
- layer 4 can identify dog faces, bird legs

- first we only added some layers
- for fine tuning we will change all layers before and check if we can improve them
- some early layers should not be changed, but the last ones seems likely to be changed
- learning rate finder will check the fastest steps for learning rate

- all of the doc is a jupyter notebook and you can clone it and try things out
