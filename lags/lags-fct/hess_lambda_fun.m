function [hess_lambda] = hess_lambda_fun(x, c_k, att_k)

[M,N] = size(x);

% Exponentially decaying function for 
att_k_ = repmat(att_k,[1 size(x,2)]);
diff_ = x' - att_k_';
diff_norm = sqrt(abs(sum((x' - att_k_').^2, 2)));
exp_ = exp(-c_k * diff_norm.^2);    
%     scalar_factor = (c_k * exp_./diff_norm);    
scalar_factor = (2* c_k * exp_);

hess_lambda = zeros(M,M,N);
for n=1:N   
    x_diff = diff_(n,:)';
    hess_lambda(:,:,n) =  scalar_factor(n)*(-2*c_k*(x_diff)*x_diff' + eye(M));
end

end