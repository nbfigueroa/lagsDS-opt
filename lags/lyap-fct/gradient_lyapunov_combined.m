function [grad_lyap] = gradient_lyapunov_combined(x, att_g, att_l, P_global, P_local)

% Variables
[N, M]    = size(x);
grad_lyap = zeros(size(x));
[~, K]    = size(att_l);    
for i = 1:M
    % Gradient of global component
    grad_global_comp = (P_global' + P_global)*(x(:,i) - att_g);
    
    % Gradient of local components
    grad_local_comp = 0;
    for k=1:K
        % Local Lyapunov Component
        lyap_local_k =   (x(:,i) - att_g)'*P_local(:,:,k)*(x(:,i) - att_l(:,k));
        
        % Computing activation term
        if lyap_local_k >= 0
            beta = 1;
        else
            beta = 0;
        end
        beta_k_2 = 2 * beta * lyap_local_k;
        grad_local_comp_k = P_local(:,:,k)*(x(:,i) - att_l(:,k)) + P_local(:,:,k)'*(x(:,i) - att_g);
        
        % Sum of Local terms
        grad_local_comp =   grad_local_comp + beta_k_2 * grad_local_comp_k;
    end
    
    % Computing full derivative
    grad_lyap(:,i) = grad_global_comp + grad_local_comp;
end

end
