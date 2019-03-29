function [f,grad_f] = fQ_functionwithGrad(x, att_g, att_l, P_g, P_l, alpha_fun, grad_alpha_fun, h_fun, grad_h_fun, A_g, A_l, A_d)
f      = -fQ_constraint_single(x, att_g, att_l, P_g, P_l, alpha_fun, h_fun, grad_h_fun, A_g, A_l, A_d);
if nargout > 1 % gradient required
    grad_f = -gradient_fQ_constraint_single(x, att_g, att_l, P_g, P_l, alpha_fun, grad_alpha_fun, h_fun, grad_h_fun, A_g, A_l, A_d);
%     if nargout > 2 % hessian required
%         hess_f  = eye(N,N);
%     end
end






end

