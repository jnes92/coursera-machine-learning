# Fast AI Part 1 

- stopped 2018 version after lesson 2
- 7 week course 
- Samstag 11.05. - Samstag 29. Juni

## Lesson 1: Samstag 11.5.19 (17 Uhr - 20 Uhr)
- 1h getting started 
- 2h video material
- 1h 30m ResNet 50, Deploy & Export files
- 1h Lesson 1 with own data: 
  - Aikido 
  - BJJ 
  - Kickboxen  
  - Krav Mage
  - Wing Chun


### Getting started

To get started you need:
1. Python (1 year coding exp)
2. GPU & Software

-> more reading here: 
https://www.fast.ai/2017/11/16/what-you-need/

- [Notes: What you need to do deep learning](./lesson1/01-what-you-need.md)

#### Learning Python

https://forums.fast.ai/t/recommended-python-learning-resources/26888

#### Using a GPU
- nvidia gpu required
- dont setup a computer right now (takes time and energy)
- rent a computer that has everything you need preinstalled
- shutdown after you are done !

recommended options:
- google cloud platform (300$ free credit)
- Salamander 
- Colab is free, minor rough edges and incompatibilites

**Ready to run: 1 Click Jupyter**
- Salamander.ai
- Google Colab https://course.fast.ai/start_colab.html
- Paperspace Gradient
- SageMaker
- Kaggle Kernels
- Floydhub
- Crestle


**Ready to run: 1 Click Jupyter**
- Google Cloud Plattform
- Azure https://course.fast.ai/start_azure.html


#### Installing in Google Colab
- sign in to google
- start colab
- search for 'fastai/course-v3'
- open a notebook (lesson1)
- get gpu -> Runtime Tab -> Change runtime type : GPU -> Ssave

- configure notebook with packages
- add code 
```  !curl -s https://course.fast.ai/setup/colab | bash
```

- save copy of the notebook to google drive / github
- data files will not be persisted
- need to permit colaboratory to rw files to goole drive with this snippet
```
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
root_dir = "/content/gdrive/My Drive/"
base_dir = root_dir + 'fastai-v3/'
```

Ready to go !

### Links

- [Video Notizen](./lesson1/02-video-image-classifier.md)
- [Convolutional Neural Networks (CNNs / ConvNets)](lesson1/03-more.md)
- [Splunk and Tensorflow for Fraud Detection](lesson1/03-more.md)


### Todo Lesson 1
[x] run the notebook
[x] test with resnet50 -> gpu memeroy - batch size decrease
[x] Create a model that can correctly classify images. Use lesson2-download.ipynb as a means of guidance. Remember, the more you practice the better you will get!

[x] checkout cnn link http://cs231n.github.io/convolutional-networks/
[x] splunk and tensorflow for security blogpost
[ ] checkout the forum and get involved?
[ ] Do export/ hosting tutorial for cat breeds
[ ] Do export/ host with own dataset


### Reminder:
- after few lessons checkout PyTorch Tutorial by Jeremy:
https://pytorch.org/tutorials/beginner/nn_tutorial.html
- do vision for person identification by myself ?
- test easy things in tensor flow 2.0 ? getting started?
- poem generator 
- GAN IMAGE Art
- Reinforcement learning test: NEAT
- Reinforcement learning test: [openAI Gym](https://gym.openai.com/docs/)
    - maybe test with toy_text: BlackJack