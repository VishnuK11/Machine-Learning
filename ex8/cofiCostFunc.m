function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

J=0;    
% For loop method here. Vector Method below
% for i=1:num_movies
%     for j=1:num_users
%         error=(X(i,:)*Theta(j,:)'-Y(i,j));
%         J=J+0.5*error*error*R(i,j);
%     end
% end

error=(X*Theta'-Y).*R;
J=sum(sum(error.*error*0.5));

% For loop method here. Vector Method below
% for i=1:num_movies
%     for k=1:num_features
%         grad_x=0;
%         for j=1:num_users
%             grad_x=grad_x+(X(i,:)*Theta(j,:)'-Y(i,j))*Theta(j,k)*R(i,j);
%         end
%         X_grad(i,k)=X_grad(i,k)+grad_x;
%     end
% end

% for j=1:num_users
%     for k=1:num_features
%         grad_t=0;
%         for i=1:num_movies
%             grad_t=grad_t+(X(i,:)*Theta(j,:)'-Y(i,j))*X(i,k)*R(i,j);
%             
%         end
%         Theta_grad(j,k)=Theta_grad(j,k)+grad_t;
%      end
% end

for i=1:num_movies
    idx=find(R(i,:)==1);
    Thetatemp=Theta(idx,:);
    Ytemp=Y(i,idx);
%     size(X(i,:))
%     size(Thetatemp')
%     size(Y)
    X_grad(i ,: ) = (X(i,:)*Thetatemp' - Ytemp)*Thetatemp+lambda*X(i,:);
end

for j=1:num_users
    idx=find(R(:,j)==1);
    Xtemp=X(idx,:);
    Ytemp=Y(idx,j);
    Thetatemp=Theta(j,:);
%     size(Xtemp)
%     size(Thetatemp')
%     size(Ytemp)
    Theta_grad(j,:) = (Xtemp*Thetatemp' - Ytemp)'*Xtemp+lambda*Theta(j,:);
end


% Regularization

J=J+0.5*lambda*(sum(sum(X.*X))+sum(sum(Theta.*Theta)));
% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
