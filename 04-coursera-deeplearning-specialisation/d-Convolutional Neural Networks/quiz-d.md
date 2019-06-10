# Week 1 - Convolutional Neural Networks


1. vertical edges
2. 27M 100
3. 2500x3 + 100 bias = 7600
4. 29x29x32 ($\frac{n+2p-f}{s} +1$)
5. 15x15x8 + 2 pad = 19x19x8
6. 3,5 downwards - 3
7. max pooling with 2x2 = half w,h both -> 16x16x16
8. false 
9. reduces params, allows to be used locations, shared for other tasks?? 
10. depend on only small number from prev. layer

# Week 2 - Deep convolutional models

1. nh,nw decrease, nc increase
2. multi conv - followed pool, fc in last few layers
3. false
4. plain: false
5. al and 0 ??
6. **WRONG**
X L^2, skip to identity mapping
X easy to learn identity mapping, computes non-linear
7. 16+1 = 17
8. 1x1 conv: reduce nc, not nh,nw + pooling reduce nh,nw not nc
9. use 1x1 conv before, single block allows combination
10. get implementation, parameters useful

# Week 3 - Detection Algorithm

1. y = 1, 0.3 , 0.7 , 0.3, 0.3, 0,1,0
2. y= 0, ??????????
3. logistic unit, bx, by
4. 2N 
5. false
6. false
7. true
8. 1/9
9. 5
10. 19x19x(5x25), with 5 anchor boxes: 25 = found sth, box, classes