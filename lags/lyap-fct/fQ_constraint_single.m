function [f_Q] = fQ_constraint_single(x, att_g, att_l, P_global, P_local, alpha_fun, h_fun, grad_h_fun, A_g, A_l, A_d)

[N,M]    = size(x);

% Variables
alpha  = feval(alpha_fun,x);
h      = feval(h_fun,x);

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

% Output Variable
f_Q = zeros(1,M);
for i = 1:M        
            
    %%%%%%%%%%%%% COMPUTING AUX VARIABLES FOR LYAPUNOV GRADIENT %%%%%%%%%%%%
    % Local Lyapunov Component
    lyap_local =   (x(:,i) - att_g)'*P_local*(x(:,i) - att_l);       
               
    % Computing activation term
    if lyap_local >= 0
        beta = 1;
    else
        beta = 0;
    end
    beta_l_2 = beta*2*lyap_local;

    %%%%%%%%%%%%%%% COMPUTING TERMS FOR STATE DYNAMICS %%%%%%%%%%%%%%%
    % Compute local dynamics variables
    if h(i) >= 1
        h_mod = 1;
    else
        h_mod = h(i)*h_set;
    end
    A_L = h_mod*A_l + (1-h_mod)*A_d;        
    
    % Grouped Matrices
    Q_g  = A_g'*P_global + P_global*A_g;
    Q_gl = A_g'*P_local;
    Q_lg = A_L'*(2*P_global);
    Q_l  = A_L'*P_local;

    % Computing Block Matrices
    Q_G = alpha(i) * ( Q_g + beta_l_2*Q_gl );
    Q_LG = (1-alpha(i))*( Q_lg + beta_l_2*Q_l );   
    Q_GL = beta_l_2*alpha(i)*Q_gl;
    Q_L  = (1-alpha(i))*beta_l_2*Q_l;

    % Block Matrix
    xi_aug  = [x(:,i) - att_g; x(:,i) - att_l];
    Big_Q   = [Q_G Q_GL; Q_LG Q_L];
   
    % LMI format of Lyapunov Derivative
    f_Q(1,i) = xi_aug'*Big_Q*xi_aug;    
end

end
