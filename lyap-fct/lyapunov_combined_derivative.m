function [lyap_der] = lyapunov_combined_derivative(x, att_g, att_l, ds_lin, lyap_type, varargin)

[N,M]    = size(x);

% Variables
xd       = feval(ds_lin,x);
lyap_der = zeros(1,M);
[N, K] = size(att_l); 
if K == 1
    for i = 1:M
        switch lyap_type
            case 0
                grad_global_comp = (x(:,i) - att_g);
                grad_local_comp_g = 2*(x(:,i) - att_g)'*(x(:,i) - att_l)*(x(:,i) - att_l);
                grad_local_comp_l = 2*(x(:,i) - att_l)'*(x(:,i)-att_g)*(x(:,i) - att_g);
                lyap_local =   (x(:,i) - att_g)'*(x(:,i) - att_l);
            case 1
                P_global = varargin{1};
                P_local  = varargin{2};
                beta_eps = varargin{3};
                lyap_local =   (x(:,i) - att_g)'*P_local*(x(:,i) - att_l);
                
                % Gradient of global component
                grad_global_comp = P_global*(x(:,i) - att_g) + P_global'*(x(:,i) - att_g);
                
                % With simplified local gradient
                grad_local_comp = 2*(x(:,i) - att_g)'*P_local*(x(:,i) - att_l)* (P_local*(x(:,i) - att_l) + P_local'*(x(:,i) - att_g));
%                 grad_local_comp = P_local*(x(:,i) - att_l) + P_local'*(x(:,i) - att_g);
        end
        
        % Computing activation term
        if lyap_local >= beta_eps
            beta = 1;
        else
            beta = 0;
        end
        
        % Computing full derivative
        lyap_der(1,i) = xd(:,i)' * (grad_global_comp + beta*grad_local_comp);
        
    end
else
%     fprintf('Multiple attractors...\n');
    P_global  = varargin{1};
    P_local   = varargin{2};    
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
        lyap_der(1,i) = (grad_global_comp + grad_local_comp)' * xd(:,i);
    end
    
end
end