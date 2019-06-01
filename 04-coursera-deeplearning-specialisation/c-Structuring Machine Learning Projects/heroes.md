# Heroes of Deep Learning - Part 3

## Week 1 - Andrej Karpathy

personal story:
- undergraduate in university
- liked the idea of mind of the network
- sth magical happening on lots of digits
- masters degree: deeper into networks
- very interesting, but not satisfying courses
- machine learning like technical term
- neural networks seems like the ai
- new computing paradise
- in-out specification can sometimes write better code than you
- new way of programming

human benchmark for imagenet:
- world cup of cv
- error rate goes down
- where is a human on this scale ?
- lowest error by humans like 6-10% baseline
- js interface showing images, with 1000 categories
- fun exercise
- organize other ppl to do the same thing (1-2 weeks)
- surprised by software deep networks surpassing you

dl class:
- research on hold
- highlight of PhD, class
- exciting students
- taught things from 1800
- paper from yesterday or week ago
- enjoyable, understood everything that happens under the hood
- forefront of sth big

evolve understanding of dl:
- cnn were around, but not used often or soon
- just for small cases, but never scales
- extreme incorrect
- no one saw coming: pre-trained network, fine tuning on other 
- general feature extractor 
- crushing each task by fine-tuning network
- hopes of unsupervised learning (2007) not delivered
- supervised learning works really well right now
- lot of deep believers for unsupervised learning

future of ai:
- openAI
- field will split into 2 categories
- applied ai: getting better, using
- artificial general intelligence: nn that can do everything a human can do
- cv was approached wrong in the beginning: breaking into parts, putting it together
- similiar on a higher level on ai
- try to decompose it again, same incorrect approach
- having a single nn, a full agent:
- esssay, short story: hypotheoretical https://karpathy.github.io/2015/11/14/ai
- imitate humans and other directions


advice for new ai:
- down to low level code
- implement chunks of it by yourself
- dont abstract, do things from scratch
- work with frameworks, after you have done it once yourself

## Week 2 - Ruslan Salakhutdinov
director or apple research

personal story: 
- started by luck
- master at toronto
- first year financial
- bumped into geoff hinton
- boltzmann machine constraing divergence
- really exciting
- started PhD with Geoff 2005 / 2006
- deep learning was popping up there

restricted bolzman machines for multiple layers
- non-linear extension of PCA
- extend to deal with faces? 
- can we compress documents?
- real value, count, binary data testing
- 6month for good results
- lot of learning with impressive results

whats happening with boltzmann machines?
- learn one layer at a time
- theoretical justification to be able to pre-tain systems
- since 2009 gpu shown up
- directly optimizing deep neural networks
- standard backprop was easy suddenly
- would have taken month on cpu in 2005
- big change
- generative models (boltzmann)
- learning algorithm marvov, monte carlo is not as scalable as backward propagation

generative unsupervised vs supervised
- important topic
- unsupervised, semi-supervised, generative models
- lot of progress was in supervised learning
- unsupervised pre training models can help for supervised models
- look at GAN, deep energy models
- not figured out: how to do unsupervised learning with some hints, some examples

supervised learning vs other approach, evolved thinking:
- we should be able to make progress there
- boltz machines, gans - generative models
- in it sector: companies have lots of unlabeld data
- we should be able to, because we have so much of it

enter dl research advice:
- try different things
- dont be afraid to try new things
- lacking theory encourages :D
- understand deep learning
- backprop for cnn, you will understand how the system operate
- how to implement on gpu
- academic xp and industry
  
PhD vs industry:
- can do amazing research in industry
- academia: more freedom to work on longterm problems
- industry is also exciting, because you can impact millions of users
- industry has more ressources for computations, ... 
- academicy -> industry: switching

exciting trending areas:
- deep reinforcement learning 
  - train agents in virtual worlds
  - lot of progress here
  - agents communicate with each other
- nl understanding
  - dialogue based system
  - read text, answer questions intelligently
- subarea: able to learn from few examples 
  - one shot learning