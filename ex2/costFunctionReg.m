function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n=size(X,2)-1;
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

reg=(sum(theta.^2)-theta(1).^2)*lambda/(2*m);
J=-(1-y).*log(1-sigmoid(X*theta))-y.*log(sigmoid(X*theta)) ;
J=sum(J)/m + reg;
grad=((sigmoid(X*theta)-y)'*X)'/m;


for i=2:n+1,
    grad(i,1)=grad(i,1)+lambda*theta(i)/m;
end

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
