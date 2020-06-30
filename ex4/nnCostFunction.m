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

% p=predict(Theta1, Theta2, X);
X=[ones(m,1) X]; %Might not be required as taken care in predict
Z2=(Theta1*X')';
A2=sigmoid(Z2);
A2=[ones(m,1) A2];
Z3=(Theta2*A2')';
A3=sigmoid(Z3);

% Change Y to a have all class binary results : m*1 - > m*10 matrix
y1=zeros(m,num_labels);
% p1=zeros(m,num_labels);
for i=1:m,
    y1(i,y(i))=1;
%     p1(i,p(i))=1;
end
y=y1;
% p=p1;
% p(1:10,:)
% y(1:10,:)

for i=1:m,
    J=J +  (1-y(i,:))*(log(1-A3(i,:))')+ y(i,:)*(log(A3(i,:))');
end
J=-J/m;

%
% Regularization
%
T1=Theta1(:,2:end);
T2=Theta2(:,2:end);
reg=(T1(:)')*(T1(:))+(T2(:)')*(T2(:));
reg=reg*lambda/(2*m);
J=J+reg;
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



% 1. Set the input layer's values  to the -th training example . 
% Perform a feedforward pass (Figure 2), computing the activations  
% for layers 2 and 3. Note that you need to add a  term to ensure that 
% the vectors of activations for layers  and  also include the bias unit.
% In MATLAB, if a_1 is a column vector, adding one corresponds to a_1 = [1; a_1].
% 2.For each output unit  in layer 3 (the output layer), set  where  indicates whether 
% the current training example belongs to class  , or if it belongs to a different class. 
% You may find logical arrays helpful for this task (explained in the previous programming 
% exercise).
% 3.For the hidden layer , set 
% 4.Accumulate the gradient from this example using the following formula: . Note that 
% you should skip or remove . In MATLAB, removing  corresponds to delta_2 = delta_2(2:end).

Triangle2=zeros(size(Theta2,1),size(Theta2,2));
Triangle1=zeros(size(Theta1,1),size(Theta1,2));

for i=1:m,
    
    Delta3=A3(i,:)-y(i,:); 
    Delta3=Delta3';%10*1
    Delta2=Theta2'*Delta3; %26*10 . 10*1 -> 26*1
    Delta2=Delta2(2:end);  %25*1
    Delta2=Delta2.*sigmoidGradient(Z2(i,:))'; %25*1 . 1*25 - > 25*1
    
    M2=Delta3*A2(i,:); %10*1 . 1*26 - > 10*26
    Triangle2=Triangle2+M2; %10*26
    M1=Delta2*X(i,:); % 25*1 . 1*401 -> 25*401
    Triangle1=Triangle1+M1; %25*401
    Theta1_grad=Triangle1/m; %25*401
    Theta2_grad=Triangle2/m; %10*26
      
end


%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
T1=Theta1(:,2:end);
T2=Theta2(:,2:end);
reg1_grad=lambda*T1/m; %25*400
reg1_grad=[zeros(size(T1,1),1) reg1_grad];
reg2_grad=lambda*T2/m; %10*25 
reg2_grad=[zeros(size(T2,1),1) reg2_grad];

Theta1_grad=Theta1_grad+reg1_grad; %25*401
Theta2_grad=Theta2_grad+reg2_grad; %10*26 
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
