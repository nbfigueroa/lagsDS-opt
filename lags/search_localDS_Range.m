function [A_l, b_l, A_d, b_d, num_grid_violations,new_kappa]= search_localDS_Range(est_params, Data_k, A_g,  att_g, att_l, P_g, P_l, local_basis, alpha_fun, h_fun, grad_lyap,grad_h_fun, activ_fun, Mu, Sigma, lambda_fun)

% Decode input variables for estimation
A_l_prev = est_params.A_l;
b_l_prev = est_params.b_l;
A_d_prev = est_params.A_d;
b_d_prev = est_params.b_d;
kappa_min =  est_params.kappa_min;
kappa_max =  est_params.kappa_max;
num_grid_violations = est_params.current_violations;
new_kappa = kappa_max;
rate_down     = 0.1;
rate_up       = 1.1;

% Decode input variables for ds-model


if num_grid_violations > 0
    % Search down
    stability_ensured = 0;
    while ~stability_ensured
        new_kappa = new_kappa*rate_down;
        if new_kappa  < kappa_min
            new_kappa = kappa_min;
        end
        [A_l, b_l, A_d, b_d, num_grid_violations] = estimate_localDS_search(Data_k, new_kappa, A_g,  att_g, att_l, P_g, P_l, ...
            local_basis, alpha_fun, h_fun, grad_lyap,grad_h_fun, activ_fun, Mu, Sigma,lambda_fun);
        if num_grid_violations == 0
            stability_ensured = 1;
        end
        if new_kappa == kappa_min
            stability_ensured = 1;
        end
    end
    
else
    % Search up
    stability_ensured = 0;
    while ~stability_ensured
        new_kappa = rate_up*new_kappa;
        [A_l, b_l, A_d, b_d, num_grid_violations] = estimate_localDS_search(Data_k, new_kappa, A_g,  att_g, att_l, P_g, P_l, ...
            local_basis, alpha_fun, h_fun, grad_lyap,grad_h_fun, activ_fun, Mu, Sigma,lambda_fun);
        if num_grid_violations > 0
            A_l = A_l_prev; A_d = A_d_prev;
            b_l = b_l_prev; b_d = b_d_prev;
            stability_ensured = 1;
        else
            A_l_prev = A_l; A_d_prev = A_d;
            b_l_prev = b_l; b_d_prev = b_d;
        end
    end
end
end