function [grad_lambda] = grad_lambda_fun(x, c_k, att_k)

% Exponentially decaying function for 
    att_k_ = repmat(att_k,[1 size(x,2)]);
    diff_ = x' - att_k_';
    diff_norm = sqrt(abs(sum((x' - att_k_').^2, 2)));    
    exp_ = exp(-c_k * diff_norm.^2);   
%     scalar_factor = (c_k * exp_./diff_norm);    
    scalar_factor = (2* c_k * exp_);    
    grad_lambda =  repmat(scalar_factor,[1 size(x,1)])'.* diff_'; 

end