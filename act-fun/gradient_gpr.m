function [grad_gpr] = gradient_gpr(x, model, epsilon, rbf_var)
%% Get Dimension and number of data points
X_train = model.X_train';
y_train = model.y_train';

[D,M_train]   = size(X_train);
[D,M_test]  = size(x);

% Creating indices for Kernel matrix
[index_i,index_j] = meshgrid(1:(M_train+M_test),1:(M_train+M_test));

X = [X_train, x];
K = rbf_k(index_i(:),index_j(:),X, rbf_var);
i = 1:M_train;
j = (M_train+1):(M_train+M_test);

%% Page 19, Algorithm 2.1 [http://www.gaussianprocess.org/gpml/chapters/RW2.pdf]
L       = chol(K(i,i) +  epsilon .* eye(M_train,M_train),'lower'); 
% L       = chol(K(i,i),'lower'); 
beta  = L'\(L\y_train');

% Variables for gradient
Lambda  = 1/rbf_var * eye(D,D);

grad_gpr = zeros(D,M_test);
for ii=1:M_test
    X_tilde_test = bsxfun(@plus, -X_train, x(:,ii));  
%     X_tilde_test =  (repmat(x(:,ii),[1 M_train]) - X_train)';  
    grad_gpr_test = -(Lambda^-1) * X_tilde_test * (K(M_train+ii,i)'.* beta);
    grad_gpr(:,ii) = grad_gpr_test;
end


% This is the actual prediction
% Mu      = K(j,i) * beta;
% grad_gpr = Mu;


end