#### What you need to do deep learning:

What do you need (in terms of hardware, software, background, and data) to do deep learning? 

**Hardware**
- video gaming industry is larger than film & music
- gpu got hugely improved
- matrix math for rendering graphics
- exact the same math for deep learning
- this advantage is key part for neural networks power

**GPU Differences**
- not programming gpu direclty
- use software libraries like PyTorch / Tensorflow
- GPU from Nvidia !
- Cuda & OpenCL 2 main ways for gpu programming
  - cuda: most developed, ecosystem
  - proprietary language by Nvidia
  - > best option for deep learning
- AMD wants to release platfrom called ROCm for deep learning support (under development)

**without gpu**
different services:
1. crestle:
- developed by a fast ai student
- setup cloud service with all common framewarks with gpu
- gpu usage is 59 cent per hour
- easy access
2. aws cloud instance setup
- 90 cent per hour
- azure is also possible
3. build your own box:
- get some nvidia gpu fpr 300$

**software needed**
*whatever you learn today will be obsolete in a year or two*
- python is most commonly used language
- TensorFlow (google)
- PyTorch (facebook)
- MxNet (University Washington - Amazon)
- CNTK (Microsoft)
- Caffe2, DeepLearning4J. NNabla, PaddlePaddle, Keras

**dynamic vs static graph computation**
- speed for experiments and iterations are prioritized
- PyTorch supports dynamic computation
- one distinct for dl libraries is dynamic / static computations
- some support both

- **dynamic computation:** 
  - program is executed in the order you wrote it
  - easier debuggin
  - straightforward to translate idea to code
- **static compuation**
  - build a structure for your neural network in advance
  - then execute operations on it
  - compiler can do greater optimizations
  - more of a disconnect between what you intended to be & what compiler exectues

- google TensorFlow mostly uses a static computation graph
- facebook PyTorch uses dynamic computation

- fast ai chooses pytorch for the course
  - easier to debug
  - dynamic computation is better fit for natural language processing
  - traditional oop style
  - tensorflow has some confusing conventions (scope / session)

- tensorflow is the most known framework from outside

**what you need for production: not a gpu**
- in production cpu and webserver of your choice
  - you rarely need to train a model in production
  - even if you want to update your model, you dont need training
  - use webserver you like the most
  - gpu is only a speed up if you batch your data
  - 32 req / second -> GPU would be slower

**the background you need: 1 year of coding**
most deep learning materials are 
- shallow, high level and wont help you to create state-of-the-art models
- highly theoretical and assumes graduate level math background

- fast ai part 1 only requires 1 year of programming experience
- background can be in any language
- maybe learn some pythong before starting this course
- math concepts get introduced when needed

**data you need**
- you dont need google sized data for dl
- power of transfer learning makes it possible for people to apply pre-trained models to smaller datasets
