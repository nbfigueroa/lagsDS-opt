function [lambda] = lambda_mod_fun(x, breadth_mod, att_l, grad_h_fun, grad_lyap_fun)

lambda = 1 - my_exp_loc_act(breadth_mod, att_l, x);
[N, M] = size(x);

% For now
% for m=1:M
%     gamma_cond = (grad_h_fun(x(:,m))'*grad_lyap_fun(x(:,m)))/(norm(grad_h_fun(x(:,m)))*norm(grad_lyap_fun(x(:,m))));
%     if  gamma_cond < 0 || isnan(gamma_cond)
%         lambda(m) = 0;
%     end
% end
end