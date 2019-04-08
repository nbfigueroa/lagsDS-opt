function [f_Q] = fQk_constraint_multi(x, att_g, att_l, alpha_fun, h_fun, A_g, A_l, A_d, grad_lyap, grad_h_fun, lambda_fun)

[N,M]    = size(x);

% Variables
alpha     = feval(alpha_fun,x);
h         = feval(h_fun,x);
grad_V    = feval(grad_lyap,x);
grad_h    = feval(grad_h_fun, x);
lambda    = feval(lambda_fun, x);

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
else
    h_set = 1;
end

% Check if it's going against the grain
if angle_n > pi/2 || angle_n < -pi/2
    h_set = 0;
    corr_scale = 5;
    corr_scale = 0.25;
    corr_scale = 1;
else
    h_set = 1;
    corr_scale = 1;
end



% Output Variable
f_Q = zeros(1,M);
for i = 1:M        
            
    %%%%%%%%%%%%%%% COMPUTING TERMS FOR STATE DYNAMICS %%%%%%%%%%%%%%%
    delta_g = x(:,i)-att_g;
    delta_l = x(:,i)-att_l; 
    
    % Compute local dynamics variables
    if h(i) >= 1
        h_mod = 1;
    else
        h_mod = h(i)*h_set;
    end
    A_L = h_mod*A_l + (1-h_mod)*A_d ;       
    modulation_term = corr_scale*lambda(i)* grad_h(:,i)';
    fk_q = alpha(i)*(delta_g'*A_g') + (1-alpha(i))*(delta_l'*A_L' - modulation_term);
    Vk_q = fk_q*grad_V(:,i);    
    f_Q(1,i) = Vk_q;    
end

end
