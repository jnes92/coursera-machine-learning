function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%


% J(theta) = 1/m * sigma(i=1 to m) of sigma(k=1 to K) of [
%                     -y(i)k * log((h(theta)(x(i))k) -
%                    (1y(i)k) * log(1 - (h(theta)(x(i))k))]
% where h(theta(x(i))k) is the result of the NN forward prop
% and K is the total number of possible layers.


% 1.3 Forward propagation and cost function (unregularized)
% forward propagation, similar to ex3 - predict:

a1 = [ones(m,1) X];

z2 = a1 * Theta1';
a2 = [ones(size(z2), 1) sigmoid(z2)];

z3 = a2 * Theta2';
a3 = sigmoid(z3);

h = a3;

% Matrix of all classes -> (m x num_labels)
tmp = eye(num_labels);
Y_k  = (tmp(y,:));

% compute the cost
% INNER sum ouput cost per each y_k_i and h_theta_k(i) ie. per each sample
% on a separate line
% OUTER sum calculates weighted cost across all examples i.e. sums all samples
% and weights them over the number of samples
J = -(sum(sum((Y_k .* log(a3) + (1 - Y_k) .* log(1 - a3)), 2)))/m;



% 1.3 Forward propagation and cost function (with regularization)
% For this we can steal some of the logic from ex2 costFunctionReg.m

% First, we toss the first columns of each Theta(i) matrix.

Theta1Reg = Theta1(:,2:size(Theta1,2));
Theta2Reg = Theta2(:,2:size(Theta2,2));

% Now implement the regularization formula described on page 6 of ex4.

Reg = (lambda/(2*m)) * (sum(sum( Theta1Reg.^2 )) + sum( sum( Theta2Reg.^2 ) ));

% Now just add the regularization term to the previously calculated J

J = J + Reg;


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

X = [ones(m,1) X]; # used later again, so init here

% Note that this time we do not transpose a3 to create h as to make
% the following matrix multiplication slightly simpler

% Now we tranform the y result vector into a matrix where 1s in the
% columns map to the corresponding values of y

yMatrix = zeros(num_labels, m);

for i=1:num_labels,
    yMatrix(i,:) = (y==i);
endfor


for k = 1:m,
    % Step 1: Set Values and feed forward to layers 2,3
    x_t = X(k,:);
    y_t = Y_k(k, :);
    
    % Forward propagation for (x_t, y_t)
    a1 = x_t;
    z2 = Theta1 * a1';

    a2 = sigmoid(z2);
    a2 = [1 ; a2];     % add bias to vector a2

    % Now we have our final activation layer a3 == h(theta)
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);

    % Step 2: compute output error
    sigma3 = a3 - yMatrix(:,k);
         
    % Step 3: compute hidden layer error
    % Re-add a bais node for z2
    z2 = [1 ; z2];
    sigma2 = (Theta2' * sigma3) .* sigmoidGradient(z2);
    % Strip out bais node from resulting sigma2
    sigma2 = sigma2(2:end);

    Theta2_grad = (Theta2_grad + sigma3 * a2');
    Theta1_grad = (Theta1_grad + sigma2 * a1);

endfor;

% Step 5: Divide accumulated gradients by 1/m
% Now divide everything (element-wise) by m to return the partial
% derivatives. Note that for regularization these will have to
% removed/commented out.


 Theta2_grad = Theta2_grad ./ m;
 Theta1_grad = Theta1_grad ./ m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
