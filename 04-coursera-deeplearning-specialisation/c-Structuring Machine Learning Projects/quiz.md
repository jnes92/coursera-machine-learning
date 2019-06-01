#
## Week 1 - Bird recognition in the city of Peacetopia (case study)


- 10 mio dataset
- bird (1) or no bird (0) on the image

1. yes, three evaluation metrics dont make it harder, you just need to define some common metric, which has some weights to each one.
2. best accuracy under 10s & <10MB: 98%
3. accuracy: optimizing, others satisficing
4. structure: train 9,5 mio.
5. false, you can add
6. different distribution, not reflecting
7. no, not enough info? 
X yes, because bias higher than variance
8. 0.3
9. can be better than humans, but never than bayes
10. high avoidable bias (1.9): bigger model & decrease reg
11. overfit dev, get bigger dev set
12. bayes is $leq$ 0.05, now harder
13. rethink ??
 X take accuracy and false negative rate during development
14.  NOT X.
X or data augmentation 
X add and shuffle 
16. two weeks will limit speed, use 10% for experiments, buying pc power


## Week 2 - Autonomous driving (case study)

1. start with basic model
2. false (softmax would be if only 1 output is valid)
3. 500 images with mistake
4. false, you can still use it, just dont sum it up if value is missing
5. 980 + 20 for dev test
6. avoidable bias of 8.3, data-mismatch
7. / not enough info?
X right 
8. false, use data augmentation
9. 2.2: maximum improve
10. capturing subset
11.  correct in test set, not correct train set
12.  pre-trained on your dataset
13.  nothing seems promising
14.  false, A would be e2e
15.  large training set