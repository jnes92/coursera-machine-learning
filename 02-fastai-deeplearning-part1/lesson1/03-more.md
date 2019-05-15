# Convolutional Neural Networks (CNNs / ConvNets)
http://cs231n.github.io/convolutional-networks/


- similiar to Neural Network
- made up of neurons with leranable weights and biases
- each receive some input matrix multiplication
- whole network expresses a single score function: raw image pixels -> class scores 
- loss function on the last layer 

- difference to nn: ConvNet architecture make the assumption that input = image
- enables to encode certain properties
- more efficient forward function, reduce amount of parameters

Architecture Overview:
Recall NN:
- receive an input (single vector) and transform it through series of hidden layers
- each hidden layer is a set of neurons, all connected to previous layer
- last fully connected layer is called output layer and represents class scores

- regular neural networks (RNN) dont scale well to full images
- even CIFAR-10 (32x32x3 (RGB)) would result in 3072 weights fur 1 fully connected neuron
- this amount is okay, but wont scale to larger images (200x200x3 => 120.000 weights)
- full connectivity is wasteful and the huge number of params would quickly lead to overfitting

3D Volumes of Neurons
- cnn take advantage of the fact that input = images => constrain architecture
- cnn has neurons arranged in 3dimensions: width, height, depth (not NN depth)
- eg. CIFAR-10 has dimension 32x32x3
- neurons in one laer will only be connected to a small region of the layer before (not fully connected)
- final layer for CIFAR 10 would be 1x1x10 dimensional 
- convNet will reduce the full image into a single vector of class scores

![visual](./cnn_neural_net.jpeg)

Layers used to build ConvNets
- 3 main layers to build cnn: 
    - convolutional Layer
    - Pooling Layer
    - Fully-Connected Layer (like rnn)
- stack these layers to form a cnn architecture
- Example for CIFAR-10: [INPUT - CONV - RELU - Pool - FC]


# Splunk and Tensorflow for Security: 
## Catching the Fraudster with Behavior Biometrics

[Link to article](https://www.splunk.com/blog/2017/04/18/deep-learning-with-splunk-and-tensorflow-for-security-catching-the-fraudster-in-neural-networks-with-behavioral-biometrics.html)

- steady growth of cyberattacks
- primaray ways of fraud: user account takeover, credentials theft, online payment takeover 
- common detections like geo location / ip adress are known for attackers
- identifying malicious actors is becoming more challenging

- humans use computer systems in a unique way 
- mouse and other input is person´s system
- those habits and behaviors are difficult to change
- identify useres by typical patterns -> detect anomalies

- can we recongnize the user by input devices ? 
- field is called "behavioral biometrics"
- common algorithm will just learn on a subset of data 
- splunk has a software to detect each mouse movement with timestamp and x,y coordinates
- some mouse interaction may contain 5-10k events per user per page


- all those datapoints get converted into an image path
- 1 image can represent thousands of data points
- used inside a cnn with TensorFlow + Keras 
- Keras allows image recognition architecture in 50 - 100 lines of python
- gpu can speed up 10-100x of cpu calculation
- using real usage data from real web visitors from financial information service
- a scheduled python script generates images from mouse activity encoded

### Group classification

- first task was to seperate between: regular customers and new visitors for financial sites
- VGG 16 architecture for image recognition
- VGG 16 is optimized for non-natural images
- first result with 2800 images -> 81% accuracy

### Individual user classification

- second task: recognize indiidual user
- serveral extra difficulties for realistic scenario
  - small dataset: 360+180 
  - dataset was created with other people trying to achieve similiar activity
- also got 78% accuarcy