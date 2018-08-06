function [grad_lyap] = gradient_lyapunov(x, attractor, att_l, P_global, P_local)

[N,M]    = size(x);

% Output Variable
grad_lyap = zeros(N,M);

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
    grad_lyap(:,i) = grad_global_comp + beta*grad_local_comp;
    
end

end