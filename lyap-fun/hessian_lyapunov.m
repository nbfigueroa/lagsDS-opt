function [hess_lyap] = hessian_lyapunov(x, att_g, att_l, P_global, P_local)

[N,M]    = size(x);

% Output Variable
hess_lyap = zeros(N,N,M);

for i = 1:M
    
    % Auxiliary variables
    x_g        = x(:,i) - att_g;
    x_l        = x(:,i) - att_l;
    lyap_local = x_g'*P_local*x_l;
    
    % Computing activation term
    if lyap_local >= 0
        beta = 1;
    else
        beta = 0;
    end
        
    % Hessian of global component
    hess_global_comp = (P_global +  P_global');
    
    % Hessian of local component
    hess_local_comp = (P_local*x_l + P_local*x_g)*(x_g'*P_local) + ...
                      (P_local*x_l + P_local*x_g)*(x_l'*P_local) + ...
                      2*lyap_local*P_local;    
    
    % Full Hessian
    hess_lyap(:,:,i) = hess_global_comp + 2*beta*hess_local_comp;
    
end

end