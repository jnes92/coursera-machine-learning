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