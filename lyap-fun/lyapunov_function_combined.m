function [lyap_fun] = lyapunov_function_combined(x,att_g, att_l, comb_type, varargin)
% Following WSAQF approachlf sh
[N,M]    = size(x);
% Variables           
lyap_fun = zeros(1,M);
[~, K] = size(att_l); 

if K == 1
    for i = 1:M
        switch comb_type
            case 0 % WSQF
                lyap_global =  (x(:,i) - att_g)'*(x(:,i) - att_g);
                lyap_local =   (x(:,i) - att_g)'*(x(:,i) - att_l);
                
            case 1 % WSAQF
                P_global  = varargin{1};
                P_local   = varargin{2};
                lyap_global =  (x(:,i) - att_g)'*P_global*(x(:,i) - att_g);
                lyap_local =   (x(:,i) - att_g)'*P_local*(x(:,i) - att_l);
        end
        beta_eps = 0;
        % Compute Activation term for extra functions
        if lyap_local >= beta_eps
            beta = 1;
        else
            beta = 0;
        end
        % Final Sum of Quadratic Functions
        lyap_fun(1,i) =  lyap_global + beta*(lyap_local)^2;       
    end
else    
    P_global  = varargin{1};
    P_local   = varargin{2};  
    gmm = varargin{3};
    
    for i = 1:M
        lyap_global =  (x(:,i) - att_g)'*P_global*(x(:,i) - att_g);
                
        lyap_local = 0; 
        for k=1:K
            % Compute term for each attractor
            lyap_local_k = (x(:,i) - att_g)'*P_local(:,:,k)*(x(:,i) - att_l(:,k));
            
            % Compute Activation term for extra functions
            if  (lyap_local_k >= 0)
                beta = 1;
            else
                beta = 0;
            end   
            
            % Add term to final local component
            lyap_local = lyap_local + beta*(lyap_local_k)^2;
        end
        
        % Final Sum of Quadratic Functions
        lyap_fun(1,i) =  lyap_global + lyap_local;
    end    
    
    
end


end