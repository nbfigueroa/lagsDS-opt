function [lyap_der] = lyapunov_combined_derivative_global(x, att_g, att_l, P_global, P_local, A_g)

[N,M]    = size(x);


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

    %%%%%%%%%%%%%%% COMPUTING TERMS FOR LYAPUNOV DERIVATIVE %%%%%%%%%%%%%%%        
    
    % Grouped Matrices
    Q_g  = A_g'*P_global + P_global*A_g;
    Q_gl = A_g'*P_local;
    
    % Computing Block Matrices
    Q_G =  Q_g + beta_l_2*Q_gl ;   
    Q_GL = beta_l_2*Q_gl;

    % Block Matrix
    xi_aug  = [x(:,i) - att_g; x(:,i) - att_l];    
    Big_Q   = [Q_G Q_GL; zeros(N,2*N)];
    
    % LMI format of Lyapunov Derivative
    lyap_der(1,i) = xi_aug'*Big_Q*xi_aug;    
end

end
