

# fast ai : Exercise 1

## Zeitübersicht: 
- 1h Setup Azure VM
- 1h 30m Video Lesson 1
- 2h Code zu Lektion 1
- 6h Code Übung 1
- 30min Notizen verbessern

**total: 11h**

## LESSON NOTES
We learn today how to classify dogs from cats. Rather than understanding the mathematical details of how this works, we start by learning the nuts and bolts of how to get the computer to complete the task, using ‘fine-tuning’, perhaps the most important skill for any deep learning practitioner. In a later lesson we’ll learn about how fine-tuning actually works “behind the scenes”.

An important point discussed is how the data for this lesson needs to be structured. This is the most important step for you to complete—if your data is not structured correctly you will not be able to train any models.


## 1. Setup der Azure VM (02.12.18)

Den grundsätzlichen Setup kann man am besten hier nachlesen:
- [Medium HowTo : Setup Azure VM](https://medium.com/@manikantayadunanda/setting-up-deeplearning-machine-and-fast-ai-on-azure-a22eb6bd6429)
- [Azure VM DeepLearning Vorlage](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-ads.dsvm-deep-learning)

Danach benötigt man die folgenden Befehle immer wieder, um sich mit der VM zu verbinden.

```
ssh -L 8888:127.0.0.1:8888 username@ip
jupyter notebooks --no-browser
``` 
- mit `-L 8888:127.0.0.1:8888` kann man einen SSH Tunnel herstellen, der den Jupyter Notebook Port auf localhost umleitet.
- Wenn man Jupyter Notebook startet bekommt man ein Token und kann sich so authorisieren.

> **Wichtig**: 
> Die DSVM von Microsoft startet direkt die Server Variante von Jupyter Notebook auf dem Port 8000. Die Schritte zum Starten von Jupyter Notebook sind also überflüssig. 
> Man erhält direkten Zugriff auf JupyterHub in dem man den Standard Linux Usernamen + Passwort in das Webinterface eingibt !!

- FAST AI installieren 
```
git clone https://github.com/fastai/fastai.git
cd fastai
conda env create -f environment.yml
conda activate fastai
```
  - ich habe zusätzlich die Bibliothek über PIP innerhalb der conda Umgebung installiert, da ich sonst Probleme hatte, die Bibliothek zu importieren (sobald ich nicht im default Ordner, also _notebooks_ war.) 
  - Alternativ kann man auch immer die conda Umgebung vorher aktivieren und danach erst Jupyter Notebook starten



## Lesson 1 Notizen: 

### Übersicht / Kurs Struktur

**Part 1: Getting Started** 
1. CNN Image Intro:  cat vs dog image classifier
2. Structured NN Intro
3. Language RNN
4. Collaborative Filtering 

**Part 2: In Depth** 

5. Collaborative Filtering In Depth
6. Structured NN Depth
7. CNN Image In Depth
8. Language RNN in Depth

**Tipps zum gesamten Kurs:**

- aim to get fast to the end
- maybe reset and watch lecture 1-7 all a 2nd or 3rd time

### Jupyter Notebook Basics:

- open source web application
- Notebook can contain code, images, markdown
- runs code in cells (* shows loading indicator)

**Jupyter Notebook Tips:**
- hit tab for "intelliSense"
- hit shift+tab to see parameters for method
- hit shift+tab twice for documentation for method (3 times for extra window)
- prefix lines with `?` to show documentation
- prefix lines with `??` to show source code 
- press `H` for a list of keyboard shortcuts 
  - learn a few each session
- ! to get bash commands inside
- Shift-Enter: Run Code from cell
- a: Insert Cell above
- b: Insert Cell below
- m/y : Change Cell to Markdown / Code

### Top-Down Approach 

- spend more time in code, income outcome
- opposite of p1esk advice on hackernews (theory, theory, theory)

### Introduction to Deep Learning
- deep learning is part of machine learning
- machine learning was heavy time investing
- better way
    - infinetely flexible function
    - all purpose param fitting
    - fast and scalable
  - -> called deep learning

### How To Lesson 1 CNN Example

**Overview. What do we need?**

1. Fast AI Library
2. Dataset with labels
3. Train Model to know cat / dogs

 #### Fast AI Library
 
 - implements all best practices (authors could find)
 - experiments with new papers
 - built on top of PyTorch

PyTorch:
- Facebook deep learning framework
- heavily used in research / academia
- See also: Tensorflow (google)
 
### GPU: Why do we need one ? 

Neural Networks
- uses gradient descent to find best parameters - needs to be fast
- gpu performance is 10x better than cpu
- gpu 1080x costs 700$, while e5 costs 4k $

Options for getting VM with nice GPUs: 
- crestle.com 
- paperspace
- google CoLab
- Azure or other Cloud Providers
- Azure Notebooks (currently no gpu support) & Azure ML Studio
- Run it locally ! GPU is recommended.

-> Decided for Azure VM with NC6, see Setup

### Machine Learning Infos

Image classifier
- much more flexible than you might 
- go bot (labels current game state as winning or losing
- detect fraud with mouse movements

neural network (NN)
- support universal approximation theorem, but its not fast and scalable
- deep learning: NN with mutliple hidden layer

Historical use of ML in google
- started in 2012 - google brain foundation
- 2016 is used in almost all of google areas

Examples of ML
- quick replies with dl reading mail
- skype translator in real time
- painting neural doodles (semantic style transfer)
- cancer diagnosis for medical images: enlitic

Ideas of what to do with ML:
- fraud detection
- sales forecasting
- pricing


### CNN - Convolutional Neual Network

**1. convolution**:
- works with a kernel (matrix of n dimensions, often 3x3)
- linear operation multiplying image values (rgb) with kernel
- can be a layer in neural network

**2. add a non linear layer**
- sigmoid function was often used to non linear
- relu (rectified linear unit)
  - y = max (x,0) -> replace negatives with 0 
- linear layer to non linearity -> complex shapes ( key idea of neural network, can solve any problem)

**3. basics approach of gradient descent:**
- try to find local minimum
- imagine any x^2 function
- take any point and take derivative of the point to know direction where to get closer to minimum
- repeat steps 
$$x_{n+1} = x_n + \frac{dy}{dx_n} * \epsilon$$

- take small step sizes for learning rate $$\epsilon$$

**=> combining convolution - non linearity - gradient descent**

- tip: show images with high activation for them
- layer 1 was learned with gradient descent -> building our new kernel
- layer 2 takes this as input
- layer 3 can detect faces
- layer 5 could detect dogs, wheels, etc.


### Code Example Notes:

**learning rate:**
- `learn.fit(0,01, 3)` 0.01: is learning rate 
- how to choose learning rate
  - idea from "cycling learning rates": 
    - pick tiny learning rate, then double it with each step
    - will be slow at the beginning but will shoot out fast
    - at what point was the "best" improvement ?
    - looking for point dropping fastest
  - will stop before 100% duration, because had found (`learn.sched.plot`)
  - find highest learning rate, which is improving

**epochs**
- using mini batches to improve model
- how many do we need ?
  - as many as you like
  - if you do too much (accuracy gets worse -> overfitting )


### Task: 
- get other images
- try to get sense of learning rates, epochs for other examples
- whats inside data



##  Task: Custom Images 

### Overview of things to do / try
- [x] Setup own Virtual Machine / Local Machine with Jupyter & Fast.ai
- [x] [Read Overview: Lesson 1A Intro](https://github.com/reshamas/fastai_deeplearn_part1/blob/master/courses/dl1-v2/lesson_1a_course_intro.md)
- [x] [Get Fast AI Working in another folder]((https://github.com/reshamas/fastai_deeplearn_part1/blob/master/tips_faq_beginners.md#option-2--where-you-want))
  - [x]   install fastai 0.7 with pip inside conda fastai env
- [x] Exercise: Test CNN with own dataset (google images)
  - [ ] download images locally and transfer data to vm ([1](https://unix.stackexchange.com/questions/188285/how-to-copy-a-file-from-a-remote-server-to-a-local-machine), [2](https://it.cornell.edu/managed-servers/transfer-files-using-putty))
  - [x] download Images inside notebook to be independent.
- [x] Skip through Paper of RNet
- [x] Checkout Kaggle Competitions & Ressources
- [x] Try Kaggle Challenge from _Getting Started_ category

### Tipps:
- **Prepare**: good split for data might be: [80% / 15% / 5% (train / validation / test)](https://github.com/reshamas/fastai_deeplearn_part1/blob/master/tips_faq_beginners.md#q4--what-is-a-good-trainvalidationtest-split)
- **Evaluate**: `plot_confusion_matrix(cm, data.classes) ` for showing true positive, false negatives and so on


**Helpful resources, if you get stucked**
- [Wiki Thread Lesson 1](https://forums.fast.ai/t/wiki-thread-lesson-1/6825)
- [All in All Thread for Exercise](https://forums.fast.ai/t/lesson-1-part-1-v2-custom-images/10154/4)

### Ideas for exercise
- mac or windows pc ?
- hot dog or not 
- bjj or aikido 


###  Walkthrough Exercise 1: BJJ vs Aikido

What do we need to do ? 
- Get Images
- Prepare Data to be splitted in folders for Train, Val, Test 
- Checkout the notebook !

1. Grabbing Images
- [x] Read Thread [Getting Image Data](https://github.com/reshamas/fastai_deeplearn_part1/blob/master/tools/getting_image_data.md)
- [x] Read Thread [How To Scrape Web for Images](https://forums.fast.ai/t/how-to-scrape-the-web-for-images/7446/15)
- [x] Use [Google Images Download](https://github.com/hardikvasa/google-images-download) for downloading inside Python

```
from google_images_download import google_images_download

response = google_images_download.googleimagesdownload()
absolute_image_paths = response.download({<Arguments...>})
```

See Notebook file for exercise 1 for details on how to do the rest :) 