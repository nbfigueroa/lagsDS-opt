function [x_dot] = lags_ds_nonlinear_local(x, alpha_fun, ds_gmm, A_g, b_g, A_l, b_l, A_d, b_d, h_functor, lambda_functor, grad_h_functor, att_g, att_l, modulation, scale)

% Auxiliary Variables
[N,M] = size(x);
K = length(ds_gmm.Priors);

% Posterior Probabilities per local DS
beta_k_x = posterior_probs_gmm(x,ds_gmm,'norm');
% attractor_k = find(posterior_probs_gmm(att_g,ds_gmm,'norm')>0.99)
% beta_k_x = gamma_prob_fun(x,ds_gmm, attractor_k);

% Activation Function
alpha = feval(alpha_fun,x)';

% Local DS Parameters/Functions
h_k      = zeros(K,M);
lambda_k = zeros(K,M);
grad_h_k = zeros(N,M,K);
corr_scale_k = zeros(1,K);
d_att_k      = zeros(1,K);
h_set_k      = zeros(1,K);
% grad_lyap = feval(grad_lyap_fun,x);

for k=1:K
    % Partition and Modulation functions
    h_k(k,:) = h_functor{k}(x);
    lambda_k(k,:) = feval(lambda_functor{k},x);
    grad_h_k(:,:,k) = feval(grad_h_functor{k},x);
    
    % Check incidence angle at local attractor
    w = grad_h_functor{k}(att_l(:,k));
    w_norm = -w/norm(w);
    
    if size(b_g,2) == 1
        fg_att = A_g*att_l(:,k) + b_g;
    else
        fg_att = A_g(:,:,k)*att_l(:,k) + b_g(:,k);
    end
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
        h_set_k(k) = 0;
        corr_scale_k(k) = 5;
    else
        h_set_k(k) = 1;
        corr_scale_k(k) = 1;
    end
    
    % Check if deflection/correction is necessary
    d_att_k(k) = norm(att_l(:,k)-att_g);
end

% Output Velocity
x_dot = zeros(N,M);
for i = 1:size(x,2)
    
    % Estimate Local Dynamics component
    f_l = zeros(N,K);
    for k=1:K        
        if modulation            
            % Compute Local Dynamics component
            if h_k(k,i) > 1
                h_mod = 1;
            else
                h_mod = h_k(k,i)*h_set_k(1,k);
            end
            
            % No local deflective DS necessary
            if d_att_k(1,k) < 0.1
                local_DS_k = h_mod* A_l(:,:,k) * (x(:,i)-att_l(:,k));                
            else
                % Compute Local Dynamics components
                f_l_k = (h_mod*A_l(:,:,k) + (1-h_mod)*A_d(:,:,k))*(x(:,i)-att_l(:,k));
                
                % Sum of two components + correction
                local_DS_k = f_l_k - corr_scale_k(1,k)*lambda_k(k,i)* grad_h_k(:,i,k);
            end                                    
        else
            % Weighted local component
            local_DS_k = A_l(:,:,k)*x(:,i) + b_l(:,k);
        end        
        
%         % Testing
%         h_mod = 0.5;
%         local_DS_k = (h_mod*A_l(:,:,k) + (1-h_mod)*A_d(:,:,k))*(x(:,i)-att_l(:,k));
        
        % Weighted local component
        f_l(:,k) = beta_k_x(k,i) * local_DS_k;                
        
    end
    f_l  = sum(f_l,2);
    
    %%%%% Local Component %%%%%
    x_dot(:,i) =  f_l;
   
    
end
end



