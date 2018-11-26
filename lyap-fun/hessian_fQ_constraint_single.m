function [hess_fQ] = hessian_fQ_constraint_single(x, att_g, att_l, P_global, P_local, alpha_fun, grad_alpha_fun, h_fun, grad_h_fun, A_g, A_l, A_d)

[N,M]    = size(x);

% Variables
alpha       = feval(alpha_fun,x);
grad_alpha  = feval(grad_alpha_fun,x);
h           = feval(h_fun,x);
grad_h      = feval(grad_h_fun,x);

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
hess_fQ = zeros(N,N,M);
for i = 1:M        
            
    %%%%%%%%%%%%% COMPUTING AUX VARIABLES FOR LYAPUNOV GRADIENT %%%%%%%%%%%%
    % Local Lyapunov Component
    x_g        = x(:,i) - att_g;
    x_l        = x(:,i) - att_l;
    lyap_local = x_g'*P_local*x_l;       
               
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

    % Computing gradient of global term
    grad_global_first  =    (x_g'*(Q_g + beta_l_2*Q_gl)*x_g);
    grad_global_second =   2*Q_g*x_g + 2*beta*( 2*lyap_local*Q_gl*x_g  + ...
                           ((x_g'*P_local) + (x_l'*P_local))'*(x_g'*Q_gl*x_g));
                       
    hess_fQ_g = grad_alpha(:,i)*grad_global_first + alpha(i)*grad_global_second;
    
    
    
    
%     % Computing gradient of interaction term
%     grad_alpha_V_l = grad_alpha(:,i)*lyap_local + alpha(i)*((x_g'*P_local) + (x_l'*P_local))';
%     grad_alphabar_V_l = -grad_alpha(:,i)*lyap_local + (1-alpha(i))*((x_g'*P_local) + (x_l'*P_local))';
%     grad_lg_1   = 2*beta*(grad_alpha_V_l* (x_l'*Q_gl*x_g) + (alpha(i)*lyap_local)*((x_g'*Q_gl') + (x_l'*Q_gl))');
%     
%     grad_lg_2   = -grad_alpha(:,i)*(x_l'*Q_lg'*x_g) + 2*(1-alpha(i))*(grad_h(:,i)*(x_l'*(A_l'*P_global)*x_g) + ...
%                    h(i)*(x_g'*(P_global*A_l) + x_l'*(A_l'*P_global))'  -grad_h(:,i)*(x_l'*(A_d'*P_global)*x_g) + ...
%                    (1-h(i))*(x_g'*(P_global*A_d) + x_l'*(A_d'*P_global))' );
%     
%     grad_lg_3  = 2*beta*(grad_alphabar_V_l*(x_l'*Q_l*x_g) +  (1-alpha(i))*lyap_local*(grad_h(:,i)*(x_l'*(A_l'*P_local)*x_g) + ...          
%                  h(i)*(x_g'*(P_local*A_l) + x_l'*(A_l'*P_local))' -grad_h(:,i)*(x_l'*(A_d'*P_local)*x_g) + ...
%                  (1-h(i)*(x_g'*(P_local*A_d) + x_l'*(A_d'*P_local))' )));
%     
%     grad_fQ_lg = grad_lg_1 + grad_lg_2 + grad_lg_3;
%     
%     % Computing gradient of local term
%     
%     grad_local_1  = 2*beta*(1-alpha(i))*lyap_local;
%     grad_local_2  = grad_h(:,i)*(x_l'*(A_l'*P_local)*x_l) + h(i)*( (x_l'*(P_local*A_l)) + (x_l'*(A_l'*P_local)))' ...
%                   -grad_h(:,i)*(x_l'*(A_d'*P_local)*x_l) + (1-h(i))*((x_l'*(P_local*A_d)) + (x_l'*(A_d'*P_local)))';    
%     grad_fQ_l     = grad_local_1*(x_l'*Q_l*x_l) + lyap_local*(1-alpha(i))*grad_local_2 ;
    
    % Sum of all terms
    hess_fQ(:,i) = hess_fQ_g;    

end

end
