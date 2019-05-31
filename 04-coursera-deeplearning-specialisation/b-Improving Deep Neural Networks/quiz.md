# DLS-2
## Week 1 - Quiz 1

1. 98,1,1
2. same distribution
3. 
   1. high bias -> increase units, layers
   2. high variance -> regularize, more train data
4. good training error, poor dev error -> high variance -> increase regularization, more data, bigger network
5. weight decay: A Regularization technique
6. lambda up -> weights down
7. you dont apply drouput, do not keep factor used in training
8. reduc regul, lower training error
9.  data augment, dropout, l2 reg
10. cost function faster

## Week 2 - Quiz 2:
1. (): example, [] layer, {}: mini, batch
2. one iteration of mini batch is faster than one iteration of batch gradient descent
3. if mini batch size is m -> batch gradient descent, if its 1 you lose vectorization
4. if you use mini batch : acceptable
5. 7.5, 10 -> $v_2 = \beta v_{t-1} + (1- \beta) \Theta_t = 0,5 * (0,5*10°) C + 0,5*10$
6. $e^t$, because e is bigger than 1
7. increase -> shift red line, decrease: more oscilattion
8. (1) is GD, 2 is small b, 3 is big b (0.9 is best)
9. tuning a, adam, mini batch, better initialize
10. adam **should not** be used with batch gradient descent

## Week 3 - Quiz 3
1. use random values
2. some hyperparams are more important (like alpha, beta)
3. amout of computational power
4. $beta = 1-10**(-r-1)$ 
5. false, repeat occasionally
6. zL
7. avoid division by 0
8.  can be learned with all, set mean and variance
9. perform norm, use exponentaially weighted across mini batches from training
10. allows to write fewer lines, open source + good governance