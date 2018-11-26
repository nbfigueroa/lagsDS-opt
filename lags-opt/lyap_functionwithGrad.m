function [f,grad_f, hess_f] = lyap_functionwithGrad(x, att_g, att_l,  P_g, P_l)

f      = lyapunov_function_combined(x, att_g, att_l, 1, P_g, P_l);

if nargout > 1 % gradient required
    grad_f = gradient_lyapunov(x, att_g, att_l, P_g, P_l);
    if nargout > 2 % hessian required
        hess_f = hessian_lyapunov(x, att_g, att_l, P_g, P_l);
    end
end
end

