# Week 2 Quiz:

1. Neuron computes a linear function followed by a activation function like sigmoid, ReLu
2. Logistic Loss is $-(y^i \log \hat{y})+(1-y^i) \log(1-\hat{y}^i))$
3. Reshapre 32,32,3 array to column vector `img.reshape(32x32x3,1)`
4. addition of shape (2,3) + (2,1) should return a (2,3) shaped matrix
5. multiply (4,3) * (3,2) = ERROR
6. dimension of X = **n_x m?** // not (1,m)
7. matrix multiply (12288,150) x (150,45) = (12288,45)
8. **a+b.T** // not:  a.T + b or
9. invoke broadcasting, element-wise, shape (3,3)
10. with 
- $J = u+v-w$
- $u = ab$
- $v = ac$
- $w= b+c$

$J = (ab) + (ac) - (b+c)$
$J = (a-1) * (b+c)$

# Week 3 Quiz:

1. 
- X is matrix COLUMN as training example
- $a^{[2](12)}$ 2nd layer, 12th example
- $a^{[2]}$ 2nd layer activation
-  $a^{[2]}_4$ 2nd layer activation, 4th neuron

2. tanh better than sigmoid: True
3. zl = wl al-1 +bl, al = gl zl
4. output layer: classification sigmoid (exception case)
5. B: (4,1) 
6. initialize weights to 0 nn: never get a correct value
7. logistic regression : no need to randomly set
8. large weights random: slow training (inputs very large, gradient close to 0)
9. Shapes: b1: (4,1) W1: (4,2)  w2 (1,4) b2 (1,1)
10. Shape: 4,m

# Week 4 Quiz

1. cache: pass variables during forward to backward
2. hyperparams: #iterations, #layer, alpha, #units
3. deeper layers compute more complex features
4. for loop is needed for the layers
5. 
```python
for(i in range(1, len(layer_dims))):
  parameter[‘W’ + str(i)] = np.random.randn(layers[i], layers[i-1])) * 0.01
  parameter[‘b’ + str(i)] = np.random.randn(layers[i], 1) * 0.01
```
6. L is 4 hidden layers 3
7. forward and backprop need to know activation functions for its layers
8. Circuit theory: true
9. W1 (4,4), b1(4,1), W2(3,4) b2(3,1) W3 (1,3) b3(3,1) 
- $W^{[l]}: (n^{[l]}, ^{[l-1]}$
- $b^{[l]}: (n^{[l]}, 1)$
1.  $(n^{[l]}, n^{[l-1]})$