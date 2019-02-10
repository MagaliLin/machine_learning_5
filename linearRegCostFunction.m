function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
m = length(y);
n = length(theta);
v = ones(m, 1);
%size(v)
w = ones(1, n);
%size(X)
%X = [ones(m, 1) X];
H = X * theta;
%size(H)
J = (1 / (2 * m)) *(((H - y) .^2)' * v) + (lambda / (2 * m)) * (w(2 : n) * (theta (2: n)  .^2 )) ;
%size(J)
% 
grad = (1 / m) * (X' * (H - y));
grad(2 : n) = (1 / m) * (X' * (H - y))(2 : n) + (lambda / m) * theta(2 : n);

% =========================================================================

grad = grad(:);

end
