function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


% J(θ) = (1/m) * −y(i) log(hθ(x(i)))−(1−y(i))log(1−hθ(x(i)))
predictions = sigmoid(X * theta)

theta_cp = theta
theta_cp(1) = 0

J = (1 / m) * ((-y' * log(predictions)) - ((1 - y') * log(1 - predictions))) + ((lambda / (2 * m)) * sum(theta_cp .^ 2))

%(hθ(x(i))−y(i))x(i) -- Same that Linear, just change the hθ(x) that is changed to sigmoid
grad = ((1/m) * (X' * (predictions - y))) + (lambda / m * theta_cp)





% =============================================================

end
