function [lyap_der] = lyapunov_combined_derivative_nonlinear_global(x, att_g, att_l, P_global, P_local, A_g, ds_gmm)

[N,M]    = size(x);

gamma_k = posterior_probs_gmm(Xi_ref,ds_gmm,'norm');

K = size(A_g,3);

% Output Variable
lyap_der = zeros(1,M);
for i = 1:M        

    for k=1:K
        %%%%%%%%%%%%% COMPUTING AUX VARIABLES FOR LYAPUNOV GRADIENT %%%%%%%%%%%%
        beta_k_2 = zeros(1,K);
        for j=1:K
            % Local Lyapunov Component
            lyap_local_k =   (x(:,i) - att_g)'*P_local(:,:,1)*(x(:,i) - att_l(:,k));
            
            % Computing activation term
            if lyap_local >= 0
                beta = 1;
            else
                beta = 0;
            end
            beta_k_2(1,j) = beta*2*lyap_local_k;
        end
        %%%%%%%%%%%%%%% COMPUTING TERMS FOR LYAPUNOV DERIVATIVE %%%%%%%%%%%%%%%
        
        % Grouped Matrices
        Q_g  = A_g'*P_global + P_global*A_g;
        Q_gl = A_g'*P_local;
        
        % Computing Block Matrices
        Q_G =  Q_g + beta_l_2*Q_gl ;
        Q_GL = beta_l_2*Q_gl;
    end
            
    lyap_der(1,i) = ...;    
end

end
