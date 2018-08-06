function [lyap_constr] = necc_lyapunov_stability_constraint(x, attractor, att_l, P_global, P_local, alpha_fun, h_fun, lambda_fun, grad_h_fun, A_g, A_l, A_d)

[N,M]    = size(x);

% Variables
alpha  = feval(alpha_fun,x);
h      = feval(h_fun,x);
lambda = feval(lambda_fun,x);
grad_h = feval(grad_h_fun,x);

% Check incidence angle at local attractor
w = grad_h_fun(att_l);
w_norm = -w/norm(w);
fg_att = A_g*att_l;
fg_att_norm = fg_att/norm(fg_att);

% Put angles in nice range
angle_n  = atan2(fg_att_norm(2),fg_att_norm(1)) - atan2(w_norm(2),w_norm(1));
if(angle_n > pi)
    angle_n = -(2*pi-angle_n);
elseif(angle_n < -pi)
    angle_n = 2*pi+angle_n;
end

% Check if it's going against the grain
if angle_n > pi/2 || angle_n < -pi/2
    h_set = 0;
    corr_scale = 5;
else
    h_set = 1;
    corr_scale = 1;
end


% Output Variable
lyap_constr = zeros(1,M);

for i = 1:M
    
    % Computing activation term
    lyap_local =   (x(:,i) - attractor)'*P_local*(x(:,i) - att_l);
    if lyap_local >= 0
        beta = 1;
    else
        beta = 0;
    end
        
    % Gradient of global component
    grad_global_comp = P_global*(x(:,i) - attractor) + P_global'*(x(:,i) - attractor);
    
    % With simplified local gradient
    grad_local_comp = 2*(x(:,i) - attractor)'*P_local*(x(:,i) - att_l)* (P_local*(x(:,i) - att_l) + P_local'*(x(:,i) - attractor));    
    
    % Full Gradient
    full_grad = (grad_global_comp + beta*grad_local_comp);
    
    %%%%%%%%%%%%%%% COMPUTING TERMS FOR STATE DYNAMICS %%%%%%%%%%%%%%%
    % Compute local dynamics variables
    if h(i) >= 1
        h_mod = 1;
    else
        h_mod = h(i)*h_set;
    end
    A_L = h_mod*A_l + (1-h_mod)*A_d;
    
    
    % Computing full derivative
    lyap_constr(1,i) = alpha(i)*full_grad' * (A_g) * (x(:,i) - attractor) + ...
                       (1-alpha(i))*full_grad' * ((A_L) * (x(:,i) - att_l) - corr_scale*lambda(i)* grad_h(:,i));
    
end

end