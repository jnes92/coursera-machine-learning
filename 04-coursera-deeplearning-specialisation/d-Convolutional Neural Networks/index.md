# Deep Learning Specilisation - Course 4 / 5 
# Convolutional Neural Networks

## Week 1 
### Convolutional Neural Networks
#### 01 - Computer Vision

- enables brand new applications, that werent possible few years ago
- cv research community is creative for architecture
- inspriation from ideas of cv for other domains
- image classification / Object detection
- Neural Style Transfer for artwork
- images are usually at least 1000x1000px x3 (RGB) -> 3 mio features X
- difficult to get enough data to prevent overfitting
- memory requirement is also quite big

#### 02 - Edge Detection Example

- convolutional operation is one of the building blocks for CNNs
- how to detect edges in images ? 
  - detect vertical edges
  - detect horizontal edges
- grayscale 6x6 (x1) image
  - construct 3x3 filter / kernel 1,1,1 0,0,0 -1-1-1 
  - * : Convolution operation, in python its overloaded
  - Output 4x4 matrix
  - to compute O_11 copy filter to the matrix: 3x1+1x1+2x1-1*-1... = -5
  - for O_12: shift it one to the right (0*1+5*1+7*1 - ... ) = -4
  - right output would be another image
  - python: conv_forward, tf.nn.conv2d, ...
- vertical edge detection
  - imagine simple half white left, grey right
  - filter looks like: white left, grey middle, black right
  - output : 0 - 30 - 30 - 0, would be grey - white - grey
  - detected edge in the middle (seems thick, but only for 6x6 px :D)
  - vertical edge is a bright on the left dark on the right

#### 03 - More Edge detection

- what happens if it is switched ? grey - left halfed
- output would be 0 | -30 | -30 | 0
- can take absolutes if you dont care, just detect lines
- 1 0 -1 | 1 0 -1 | 1 0 -1 would detect horizontal lines
- dark to bright is negative edge
- bright to dark is a positive value
- there are many filters you can use
  - Sobel filter: 1 2 1 | 0 0 0 | -1 -2 -1 
  - or Scharr filter: 3 10 3 | 0 0 0 | -3 -10 -3
- with dl you just can treat filter as parameters that can be learned
- can also detect other edges, eg. 45, 70 degrees
- even more robust than computer vision researchers


#### 04 - Padding

- recap: do convolution 6x6 with 3x3 filter = 4x4 matrix
- math: nxn * fxf = (n-f+1) x (n-f+1) 
- each filter your image shrinks
- outer corner elements like 1,1 is only represented in 1 region, others are multiple represented, throwing information from outlines away
- to fix problems you can pad the image with an additional border around the edges
- transformed 6x6 image to 8x8 -> output will be 6x6 again
- padding is 0 everytime, p: padding amount, e.g. 1 
- Output gets: $n+2p-f+1 x n+2p-f+1$
- how much to pad?
  - two valid choices: "valid" & "same" convolutions
  - valid: n x n * fxf -> n-f+1 x n-f+1
  - same: pad so output size is equal to input size: (2+2p-f+1)
  - n+2p-f+1 = n -> p = (f-1)/2
  - convention: f is usally odd, useful to have a central pixel

#### 05 - Strided Convolutions

- another piece of building block of cnn
- e.g. stride = 2
- sets filter two steps (hor. & vert.) instead of just 1 to the right
- n x n * f x f  = $\frac{n + 2p -f}{s} +1 x \frac{n + 2p -f}{s} +1$
- if not an int round down, floor(z)
- with padding p, stride s

typical math book for convolution:
- first take filter, but then flip it horizontal and vertical
- after flipping you will compute the matrixes to compute elements
- we skipped this flipping, its called "cross-correlation"
- but by convention for deep learning its called a "convolution"
- for signal processing this gets the property: (A*b)*C = A*(b*c)

#### 06 - Convolutions over Volume

- lets see how the 2d conv works in 3d 
- image is now : 6x6x3, height x width x channels (RGB)
- conv with a 3D filter 3x3x3 (for each channels)
- channels of image and filter needs to match
- output will be 4x4 (x1)
- works just like default conv (cross-correlation)
- just take the 27 numbers and multiply with the image, sum all up for output
- you can have a filter that only detects edges in one channel (others are filled with 0)
- you can also have a filter that does not care about colors and use edge filter in each channel
- 3D image x 3D filter = 2d image
- how to use multiple filters at the same time ?
  - take output from filter 1 and output from filter 2 
  - stacking them: 4x4x2 
- summary: n x n x $n_c$ * fxfx$n_c$ = $\frac{n-f+1}{s} x \frac{n-f+1}{s} x n_c'$ with $n_c'$ as # filters
- can detect multiple features parallel

#### 07 - One Layer of a Convolutional Network

example of a layer:
- take 6x6x3 with f1: 3x3x3, f2: 3x3x3
- Output will be 4x4
- Adding non linearity ReLu(o1 + $b_1$) -> Output 1 / 2 to both filtered output
- stack them to output 4x4x2
- is one layer of cnn
- before $z^{[l]} = W^{[l]} + a^{[l-1]} + b^{[l]}$ and $a^{[l]} = g(z^{[l]})$
- filter for cnn is $W^{[l]}$
- before real output: $W^{[l]} a^{[l-1]} + b^{[l]}$
- $a^{[l]}$ is the output after adding non linearity g

number of params in one layer:
- 10 filters, with 3x3x3
- per filter: 3x3x3: 27 parameters + bias = 28
- 28*10 filters = 280 params in total
- 280 params is the same, does not matter how big the image is
- CNN works for even large images, parameters keep a "small" value

summary:
- for layer l as Convolutional layer
- $f^{[l]}$ filter size
- $p^{[l]}$ padding
- $s^{[l]}$ stride
- $n_c^{[l]}$ number of filters
- Input is: $n_H^{[l-1]$ x $n_W^{[l-1]}$ x $n^{[l-1]}_c$
- Output is  $n_H^{[l]}$ x $n_W^{[l]}$ x $n^{[l]}_c$
  - $n^{[l]}_H = \frac{n + 2p -f}{s} + 1$
  - $n^{[l]}_W = \frac{n + 2p -f}{s} + 1$
- each filter is: $f^{[l]} x f^{[l]} x n_c^{[l-1]}$
- Activations $a^{[l]} : (n_H^{[l]}, n_W^{[l]}, n_c^{[l]} )$
- Weights: $f^{[l]} x f^{[l]} x n_c^{[l-1]} x n_c^{[l]}$ , with $n_c^{[l]}$: #filters in layer l
- bias: $n_c^{[l]}$ - (1,1,1,$n_c^{[l]}$)

#### 08 - Simple Convolutional Network Example

Example ConvNet:
- image 39x39x
  - $n_H^{[0]} = n_w^{[0]} = 39$
  - $n_c^{[0]} = 3$
- filter : $f^{[1]} =3, s^{[1]}=1, p^{[1]} = 0$ with 10 filters
- calculate output: $\frac{n+2p-f}{5} +1 = 37$
- output 37x37x10
  - $n_H^{[1]} = n_w^{[1]} = 37$
  - $n_c^{[1]} = 10$
- add filter 2: $f^{[2]} =5, s^{[2]}=2, p^{[2]} = 0$ with 20 filters
- output: 17x17x20
  - $n_H^{[2]} = n_w^{[1]} = 17$
  - $n_c^{[2]} = 20$
- add filter 3: $f^{[3]} =5, s^{[3]}=2, p^{[2]} = 0$ with 40 filters
- output: 7x7x40
- flatten to 1960 -> feed to logistic regression / softmax

most common types of layer in a ConvNet:
- Convolution Layer (CONV)
- Pooling Layer (POOL)
- Fully connected (FC)

#### 09 - Pooling Layers

Pooling Layer: Max pooling
- input 4x4 matrix
- output 2x2 matrix
- break input into regions, max of regions
- Hyperparameters for max pooling: f = 2, s=2
- if feature detected anywhere, keep high number, else all is small
- main reason: experiments work very well
- formula for output size is the same like conv: $\frac{n+2p-f}{s} +1$
- for more dimensions just repeat for each channel.
- max pooling is done independently for each channel

Pooling Layer: Average pooling
- instead of maximum, take averages of numbers inside region
- 1 exception: deep in a nn to collapse e.g. 7x7x1000 -> 1x1x1000

Hyperparameters
- f : filter size
- s : stride
- max or avg. pooling
- rarely p for padding, usually 0
- interesting: hyperparameters, but nothing to learn, just fixed computation

#### 10 - CNN Example

NN example:
- similiar to classical nn: LeNet-5
- input 32x32x3
- add filters with f=5, s=1 x 6 (filter number)
- CONV 1 - output 28x28x6
- add pooling layer (max, f=2, s=2)
- POOL 1 - output 14x14x6
  - sometimes CONV1+ POOL1 : Layer 1, sometimes Layer 1: CONV1, Layer 2 Pool
  - we will use CONV1+Pool1: Layer 1, count layers that have weights
- add filter f=5, s=1, x16
- CONV2 - output 10x10x16
- add pooling (max, f=2,s=2)
- POOL2 - output 5x5x16
- flatten POOL2 -> 400
- FC3: fully-connected layer (400-120 Units) with $W^{[3]}: (120,400), b^{[3]}: (120)$
- add another Layer FC4 (120-84 Units)
- add Softmax unit (10 outputs)
  
how to choose hyperparameters:
- see literature, architecture that worked before :D 
- more next week

usually patterns in nn:
- dimension width, height decrease
- channels increase 

number of parameters:
- pooling has 0 parameters
- conv layers have very few parameters
- fc have lots more (10k vs 200)
- activation size should not decrease quickly in your nn

#### 11 - Why Convolutions?

advantages:
- usually to create a fc layer from 32x32x3 -> 28x28x6 are around 14M parameters for mini image
- but with a filter its just 5x5 +1 x6 parameters 
- **parameter sharing**: feature detector for one part of an image is useful for other parts of the image
- **sparsity of connections**: output value depends only on a small number of inputs
- translation invariance: can detect cats wherever it is 

## Week 2 
### Case Studies
#### 12 - Why look at case studies?

- best way to get intuition by seeing good examples
- architecture works well on one problem, mostly be good in other problems
- Classic Networks : LeNet-5, AlexNet, VGG
- ResNet (up to 152 layer network with nice tricks)
- Inception
- ideas could be useful for own problems

#### 13 - Classic Networks

**LeNet-5:**
- recognize handwritten image (grayscale 32x32x1)
- conv layer 6 filter 5x5, s=1 -> reduces to 28x28x6
- (classical: avg, modern: max) pooling f=2, s=2 -> 14x14x6
- conv layer 5x5, s=1 -> 10x10x16 (back then no padding was used)
- avg pool f=2, s=2 -> 5x5x16 (400 unit)
- fc layer 400 -> 120 Neurons
- fc layer 120 -> 84
- final output $\hat{y}$
- has 60k parameters 
- $n_H, n_W \downarrow$, $n_c \uparrow$
- pattern in arrangement: conv pool, conv pool, fc fc, output

**AlexNet:**
- Alex Krizhevsky 2012
- input 227x227x3 (in paper: 224x224x3)
- conv:96 filters 55x55, s=4  -> 55x55x96
- max pool: 3x3, s= 2 -> 27x27x96
- conv: 5x5 same -> 27x27x256
- max pool: 3x3, s= 2 -> 13x13x256
- conv 3x3 same
- conv 3x3 same
- conv 3x3 same
- max pool 3x3, s=2 -> 6x6x256
- fc 9216
- fc 4096
- fc 4096
- softmax 1000
- similiar to LeNet, but **much** bigger
- around 60 Mio parameters
- good paper to start reading papers, easy to follow

**VGG-16**
- Simonyan & Zisserman 2015, 2015
- simpler network with just 
  - conv=3x3, s=1, same 
  - maxpool 2x2, s=2
- input: 224x224x3 -> 2x CONV with 64 filter [CONV 64] x2 -> 224x224x64
- pool layer -> 112x112x64
- [CONV 128] x2 (128 filters, 2 conv layers) -> 112x112x128
- pool -> 56y56x128
- [CONV 256] x3 -> 56x56x256
- pool, conv, pool, conv, pool again
- fc 4096, fc 4096, softmax 1000
- 138 M parameters (pretty large network)
- filters double after each step 64-128-256-512


#### 14 - ResNets

- build of residual blocks
- $a^{[l]}$ -> Linear -> ReLu (get $a^{[l+1]}$) -> Linear -> ReLu 
- $z^{[l+1]} = W ^{[l+1]} a^{[l]} + b^{[l+1]}$
- $a^{[l+1]} = g(z^{[l+1]})$ ...
- change main path for short cut / skip connection
- $a^{[l+2]} = g(z^{[l+2]} + a^{[l]})$
- passes information deeper in the network
- allows training of deeper networks
- stack residual blocks to form a resnet
- Residual Network: take plain network, add shortcuts for every 2 layers
- for plain networks: error_rate is increasing if layers increased at some time
- in theory: should be going down while adding layers
- for ResNet: error rate is constantly decreasing (> 100 layers, > 1000 layer)

#### 15 - Why ResNets Work

- if wl+2 = 0, $a^{[l+2]} \Rightarrow a^{[l]}$
- identity function is easy to learn, so will not hurt permance
- for deep plain networks: deeper and deeper gets more difficult to be able to learn identity function -> will make performance worse
- take 34 plain network and add skipped connection to get ResNet-34

#### 16 - Networks in Networks and 1x1 Convolutions

- 1x1 filter with just one number, e.g. 2
- take 6x6x1 input -> multiply with 1x1 filter
- seems not useful
- take 6x6x32, 1x1x32 filter = 6x6x#filter
- take element wise product for each layer, single real number
- like a fc nn applies to each of layer, 32 inputs -> output # filters
- can do non trivial computations
- 1x1 conv, or called *Network in Network*

- imagine 28x28x192 -> how to shrink to 28x28x32? 
- 1x1x192 with 32 filters conv 1x1
- also adds non linearity
- useful for building *inception network*

#### 17 - Inception Network Motivation

- 1x3 filter or 3x3 or 5x5 or pooling layer? 
- just add them all ?!
- input: 28x28x192
  - parallel:
  - 1x1 with 64 filter -> 28x28x64 output
  - 3x3 same -> 28x28x128
  - 5x5 same -> 28x28x32
  - max pool  with padding to match dimensions
- stack all the outputs together
- inception module output: 28x28x256
- heart of inception network by Szegedy (2014)
- you dont need to pick filter sizes, do them all, let network learn what to use
- problem: computation costs

eg. focus on 5x5 filter 
- 28x28x192 -> Conv 5x5, same, 32 -> 28x28x32 output
- 32 filters: each filter is 5x5x192
- compute 28x28x32 * 5x5*192 multiplications = **120M**

alternative using 1x1 conv
- 28x28x192 -> CONV 1x1, 16 -> Output 28x28x16
- add conv 5x5, 32 -> 28x28x32
- shrunk to intermediate volume with 16 channels *bottleneck layer*
- compute cost:
  - 1st layer: 1x1x196 * 28*28x16 filters = 2.4M
  - 2nd layer: 28x28x32 * 5x5x16 = 10.0 M
  - total: 12.4 M 
- compared to 120M -> 12.4M 
- number of additions is similiar to multiplications

#### 18 - Inception Network

- inception module: inputs previous activation, compute in parallel
  - 1x1 CONV -> 5x5 CONV -> 28x28x128 output
  - 1x1 CONV -> 3x3 CONV -> 28x28x32 output
  - 1x1 CONV 
  - MAXPOOL 3x3, same -> 1x1 CONV to shrink channels from 192-> e.g. 32
- channel concat: 64+ 128 + 32 + 32 = 28x28x256 output

inception network:
- puts a lot of inception modules together
- additional side branches added
- takes hidden layer, fc, fc, softmax to try to predict output in the end

fun facts:
- invented by Google, called "GooLeNet"
- inception meme: we need to go deeper
- meme as motiviatino for building deeper neural networks... :D 

### Practical advices for using ConvNets

#### 19 - Using Open-Source Implementation

- lot of researcher open source their work on GitHub
- rest is like a GitHub clone tutorial. :'D

#### 20 - Transfer Learning

- lots of dataset: ImageNet, MS COCO, Pascal types
- training takes several weeks and multiple gpu
- download open-source weights and use that as initilization
- e.g. cat classifier: "Tigger", "Misty", "Neither"
- training set is rather small for 2 cats
- download open-source network + weights from ImageNet with 1000 classses
- remove softmax with 1000 output -> replace with softmax 3 (Tigger, Misty, none)
- freeze parameters in all layers before softmax
- just train softmax layer
- might get good performance directly
- depending on framework: trainableParameters = 0 or freeze=1 to specify training for layers
- because all parameters are frocen the function is fixed, so you can precompute the last layer and save them to disk
- use the fixed function and compute feature vector for it

for larger dataset:
- freeze fewer layers and train the later layers, or replace later layers
- more data -> less frozen layers

for lot of data:
- just take whole thing as initiliaztion (replace random init)
- take all layers and train, replace softmax like before

> transfer learning should be likely to be done everytime


#### 21 - Data Augmentation

- for cv: more data will help almost alwasys

common augmentation:
- mirroring (horizontal)
- random cropping (not perfect, because some crops are not recognizable)
- rotation, shearing, local warping, ...

color shifting:
- add distortions to color channels
- for different light (sun, inside, outside, eary / late)
- advanced: PCA, see AlexNet paper - PCA Color augmentation, keeps overall color same

distortions during training
- data saved on harddisk
- cpu-thread 1a: loads image from hd to a stream
- cpu-thread 1b: applying distortion to mini-batch of data
- other process: training on cpu/ gpu
- can be run in parallel

#### 22 - State of Computer Vision

**data vs hand engineering:**
- spectrum between little data - lots of data
- speech recognition : decent data (like 7/10)
- image recognition: much data, but more useful (5/10)
- object detection: rather small data (2/10)
- for more data: simpler algorithm, less hand-engineering
- in contrast: more hand-engineering ("hacks")
- algorithm has 2 sources of knowledge:
  - labeled data
  - hand engineered features / network architecture / other components
- transfer learning is helpful in most cases

**tips for benchmarks/competitions:**
- most are not used for production
- ensembling:
  - train several networks independently and average their outputs
  - 3-15 networks
  - 1-2% better
- multi-crop at test time
  - run classifier on multiple versions of test images and average results
  - 10-crop: 1 central + 4 sides + mirrored 1 central + 4 sides mirror
  - could also be used for production, but not so common 

**use open source code**
- use architecture published in literature
- start with some other architecture
- use open source implementation if possible
- use pre-trained models and fine-tune on your dataset