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