function [lambda] = lambda_mod_fun(x, breadth_mod, att_l, grad_h_fun, grad_lyap_fun)

lambda = 1 - my_exp_loc_act(breadth_mod, att_l, x);
[N, M] = size(x);

grad_h    = feval(grad_h_fun, x);
grad_lyap = feval(grad_lyap_fun, x);

% For now
for m=1:M
    gamma_cond = (grad_h(:,m)'*grad_lyap(:,m))/(norm(grad_h(:,m))*norm(grad_lyap(:,m)));
    if  gamma_cond < 0 || isnan(gamma_cond)
        lambda(1,m) = 0;
    end
end
end