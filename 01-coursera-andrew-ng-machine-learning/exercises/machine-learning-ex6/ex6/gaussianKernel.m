function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% You need to return the following variables correctly.
sim = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the similarity between x1
%               and x2 computed using a Gaussian kernel with bandwidth
%               sigma
%
%
% Sum should be  = sum( (1-0)^2 + (2-4)^2 + (1 - (-1))^2 ) = sum(1+4+4) = 9 
sumOfDistancesSquared = sum((x1-x2) .^ 2); 
innerTerm = - sumOfDistancesSquared / (2 * sigma' * sigma); % for sigma = 2: -9 / (2*4) = -1.125
sim = exp ( innerTerm );




% =============================================================
    
end
