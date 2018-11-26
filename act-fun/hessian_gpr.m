function [hess_gpr] = hessian_gpr(x, model, epsilon, rbf_var)
%% Implementation of hessian of GPR.. not sure it is correct... will check with a simpler set of points
% Get Dimension and number of data points
X_train = model.X_train';
y_train = model.y_train';

[D,M_train] = size(X_train);
[D,M_test]  = size(x);

% Creating indices for Kernel matrix
[index_i,index_j] = meshgrid(1:(M_train+M_test),1:(M_train+M_test));

X = [X_train, x];
K = rbf_k(index_i(:),index_j(:),X, rbf_var);
i = 1:M_train;
j = (M_train+1):(M_train+M_test);
L       = chol(K(i,i) +  epsilon .* eye(M_train,M_train),'lower'); 
% L       = chol(K(i,i),'lower'); 
beta  = L'\(L\y_train');

% Variables for gradient
Lambda   = 1/rbf_var * eye(D,D);
I_train  = ones(1,M_train);
hess_gpr        = zeros(D,D,M_test);
for ii=1:M_test
    X_tilde_test = bsxfun(@plus, -X_train, x(:,ii));  
    
    % Computing first 'inner' term
    inner_first_term  = I_train*(K(M_train+ii,i)'.* beta);
    
    % Computing second 'inner' term
    grad_gpr_test = -(Lambda^-1) * X_tilde_test * (K(M_train+ii,i)');
    inner_second_term = X_tilde_test * (grad_gpr_test'.* beta);
    
    % Final hessian for test point
    hess_gpr(:,:,ii) = (Lambda^-1)*(inner_first_term - inner_second_term);
end

% not 100% sure this is the actual equation

end