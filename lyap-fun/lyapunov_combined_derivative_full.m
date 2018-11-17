function [lyap_der] = lyapunov_combined_derivative_full(x, att_g, att_l, P_global, P_local, alpha_fun, h_fun, lambda_fun, grad_h_fun, A_g, A_l, A_d)

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
lyap_der = zeros(1,M);
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
    
    % Auxiliary variable
    Q = P_global + P_global' + beta_l_2*P_local';
    
    %%%%%%%%%%%%%%% COMPUTING TERMS FOR LYAPUNOV DERIVATIVE %%%%%%%%%%%%%%%        
    % Computing collected terms wrt. constraints   (Unnecesary... old implementation)
%     lyap_der_global_term       = (x(:,i) - att_g)' *  (A_g'*Q)* (x(:,i) - att_g);
%     lyap_der_global_inter_term = (x(:,i) - att_g)' * beta_l_2 * (A_g'*P_local) * (x(:,i) - att_l);                               
%     lyap_der_local_term        = (x(:,i) - att_l)' * beta_l_2 * (A_L' * P_local) * (x(:,i) - att_l);                        
%     lyap_der_local_inter_term  = (x(:,i) - att_l)' * (A_L' * Q) * (x(:,i) - att_g);   
    
    % Computing modulation term
    lyap_der_grad     = (P_global + P_global')*(x(:,i) - att_g) + ...
                        beta_l_2*(P_local * (x(:,i) - att_l) + P_local' * (x(:,i) - att_g));
    lyap_der_mod_term =  corr_scale*lambda(i)* grad_h(:,i)'*lyap_der_grad;       

    
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
    lyap_der(1,i) = xi_aug'*Big_Q*xi_aug - (1-alpha(i))*lyap_der_mod_term;    
end

end

% not with squared local component
%     lyap_der_global_term       = (x(:,i) - att_g)' *  (P_global*A_g + A_g'*P_global + beta*P_local*A_g)* (x(:,i) - att_g);
%     lyap_der_global_inter_term = beta * (x(:,i) - att_g)' * (A_g'*P_local) * (x(:,i) - att_l);

% Not squared local component
%     lyap_der_local_term        = beta * (x(:,i) - att_l)' * (A_L' * P_local) * (x(:,i) - att_l);
%     lyap_der_local_inter_term  = (x(:,i) - att_g)' * (2*P_global*A_L + beta*P_local*A_L) * (x(:,i) - att_l);
% Not squared local component
%     lyap_der_mod               = (2*(x(:,i) - att_g)'*P_global + 2*beta*((x(:,i) - att_l)'*P_local + (x(:,i) - att_g)'*P_local)) ...
%                                   * corr_scale*lambda(i)* grad_h(:,i);


%%%%%%%%%%%%%%%%% Compact Way of Computing the Gradient  %%%%%%%%%%%%%%%
% Compute Gradient Components
%     grad_global_comp = P_global*(x(:,i) - att_g) + P_global'*(x(:,i) - att_g);

% With simplified local gradient
%     grad_local_comp =  P_local*(x(:,i) - att_l) + P_local'*(x(:,i) - att_g);

% Compute state dynamics
%     xd = alpha(i)*A_g*(x(:,i) - att_g) + (1-alpha(i))*A_L*(x(:,i) - att_l) - corr_scale*lambda(i)* grad_h(:,i);

% Global and Local Component of V_dot
%     Vg_dot = (grad_global_comp)'* xd;
%     Vl_dot = 2*lyap_local*(grad_local_comp)'* xd;

% Sum of Global and Local Component
%     lyap_der(1,i) = Vg_dot + beta*Vl_dot;

%%%%%%%%%%%%%%%%% Expanded Way of Computing the Gradient %%%%%%%%%%%%%%%
% Computing full derivative
% Expansion of global component ==> Vg_dot = (grad_global_comp)'* xd;
% Vg_dot = (x(:,i) - att_g)'*alpha(i)*(P_global'*A_g + P_global*A_g)*(x(:,i) - att_g) + ...
%     (x(:,i) - att_g)'*(1-alpha(i))*(P_global'*A_g + P_global*A_L)*(x(:,i) - att_l) - ...
%     (x(:,i) - att_g)'*(P_global + P_global')*corr_scale*lambda(i)* grad_h(:,i);

% Expansion of local component  ==> Vl_dot = 2*lyap_local*(grad_local_comp)'* xd;
% Vl_dot_ = (x(:,i) - att_g)'*alpha(i)*(P_local*A_g)*(x(:,i) - att_g) + ...
%     (x(:,i) - att_l)'*(1-alpha(i))*(P_local'*A_L)*(x(:,i) - att_l) + ...
%     (x(:,i) - att_g)'*(alpha(i)*P_local*A_g')*(x(:,i) - att_l) + ...
%     (x(:,i) - att_g)'*(1-alpha(i))*P_local*A_L*(x(:,i) - att_l) - ...
%     (x(:,i) - att_g)'*P_local*corr_scale*lambda(i)* grad_h(:,i) - ...
%     (x(:,i) - att_l)'*P_local'*corr_scale*lambda(i)* grad_h(:,i);
% 
% Vl_dot = 2*lyap_local*Vl_dot_;

% Sum of Global and Local Component
% lyap_der(1,i) = Vg_dot + beta*Vl_dot;


% lyap_der_global_term = (x(:,i) - att_g)' * alpha(i)*(P_global'*A_g + P_global*A_g + ...
%     beta*2*lyap_local*P_local*A_g)* (x(:,i) - att_g);
% lyap_der_local_term  = (x(:,i) - att_l)' * beta*2*lyap_local*(1-alpha(i))*(P_local'*A_L) * (x(:,i) - att_l);
% lyap_der_inter_term  = (x(:,i) - att_g)' * ((1-alpha(i))*(P_global'*A_g + P_global*A_L) + ...
%     beta*2*lyap_local*(alpha(i)*P_local*A_g' + (1-alpha(i))*P_local*A_L)) * (x(:,i) - att_l);
% lyap_der_mod_global  = (1-alpha(i))*(x(:,i) - att_g)' * (P_global + P_global' + beta*2*lyap_local*P_local)*corr_scale*lambda(i)* grad_h;
% lyap_der_mod_local   = (1-alpha(i))*(x(:,i) - att_l)' * (beta*2*lyap_local*P_local')*corr_scale*lambda(i)* grad_h;
