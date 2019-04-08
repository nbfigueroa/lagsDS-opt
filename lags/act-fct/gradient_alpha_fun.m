function [grad_alpha] = gradient_alpha_fun(x, radius_fun, grad_radius_fun, p_act_fun, grad_p_act_fun, p_type)

% Auxiliary variables
[D,M] = size(x);

% Evaluate all functions
r          = feval(radius_fun, x)';
grad_r     = feval(grad_radius_fun, x);
p_act      = feval(p_act_fun, x)';      % GPR or GMM or GAUSS
grad_p_act = feval(grad_p_act_fun, x);  % Gradient of GPR or GMM or GAUSS

% Gradient alpha as in Appendix C of lags-ds paper
switch p_type
    case 'gpr'
        % This is correct I believe
        grad_alpha = grad_r.*repmat(p_act,[D,1]) + repmat((1-r),[D 1]).*grad_p_act;
    
    case 'gmm'
        
        % Re-implement this (it is not the final version)
        Z = max(0,1 - p_act);
        grad_Z = 0.5 * (grad_p_act + repmat(sign(r),[D 1]).*grad_p_act);
        grad_alpha = repmat(1 - Z,[D,1]).* grad_r + repmat((1-r),[D 1]).*grad_Z;
    
    case 'gauss'
        p_act = p_act';
        grad_alpha = grad_r.*repmat(p_act,[D,1]) - repmat((1-r),[D 1]).*grad_p_act;
end
end