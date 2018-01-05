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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

a1 = [ones(m, 1) X]; % a¹ = x
z2 = a1 * Theta1'; % z3 = Theta1 * a¹

a2 = [ones(m, 1) sigmoid(z2)];% g(z²) (add a²0)

z3 = a2 * Theta2'; % z3 = Theta2 * a²
a3 = sigmoid(z3); % hO(x) = g(z')

%1/m * (sum(−y(i)*log((hθ(x(i))))−(1−y(i))log(1−(hθ(x(i))))))
for k = 1:num_labels % 10 labels
    %filter values for each item classification type
    yk = y == k;
    J = J + 1 / m * sum(-yk .* log(a3(:, k)) - (1 - yk) .* log(1 - a3(:, k)));
end

J = J + (lambda / (2 * m) * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2))));


%I needed of help in this part, despite to appear relative simple now
%https://github.com/merwan/ml-class/blob/master/mlclass-ex4/nnCostFunction.m
for t=1:m,
    %a1 = X(t,:);
    %2. For each output unit k in layer 3 (the output layer), set
    for k = 1:num_labels
        %δ(3) = (a(3) −yk),
        delta_3(k) = a3(t,k) - (y(t) == k); %yk
    end;
    
    %3. For the hidden layer l = 2, set δ(2) =Θ(2)T δ(3).∗gT(z(2))
    delta_2 = (Theta2' * delta_3') .* sigmoidGradient([1, z2(t,:)])';
   
    %4. Accumulate the gradient from this example using the following formula. 
    %Note that you should skip or remove δ(2) 0 . 
    %In Octave/MATLAB, removing δ(2) 0 corresponds to delta 2 = delta 2(2:end).
    delta_2 = delta_2(2:end);
    
    %∆(l) = ∆(l) + δ(l+1)(a(l))T
    Theta1_grad = Theta1_grad + delta_2 * a1(t, :);
    Theta2_grad = Theta2_grad + delta_3' * a2(t, :);
end;

Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda / m * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda / m * Theta2(:, 2:end);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
