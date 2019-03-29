%% Post-learning Numerical Stability Check using full Lyapunov Derivative
% Function for Lyap-Der evaluation
lyap_der = @(x)lyapunov_combined_derivative_full(x, att_g, att_l, P_g, P_l, alpha_fun, h_fun, lambda_fun, grad_h_fun, A_g, A_l, A_d);

% Samples to evaluate
desired_samples = 10^5;
chi_sample = [];
chi_samples = draw_chi_samples (Sigma,Mu,desired_samples,activ_fun);

% Evaluate Samples
necc_lyap_constr = lyap_der(chi_samples);

% Necessary Constraint
necc_violations = necc_lyap_constr >= 0;
necc_violating_points = chi_samples(:,necc_violations);

% Check Constraint Violation
if sum(necc_violations) > 0
    fprintf(2,'System is not stable.. %d Necessary (grad) Lyapunov Violations found\n', sum(necc_violations))
else
    fprintf('System is stable..\n')
end
if exist('h_samples_necc','var'); delete(h_samples_necc);  end
h_samples_necc = scatter(necc_violating_points(1,:),necc_violating_points(2,:),'+','r');

%% Post-learning Stability Check using Matrix inequalities/constraints
clc;
% Computing Max activation terms
if sum(necc_violations) > 0
    x_test = draw_chi_samples (Sigma,Mu,1,activ_fun);
    x_test = necc_violating_points (:,randi(sum(necc_violations)));
else           
    x_test = draw_chi_samples (Sigma,Mu,1,activ_fun);
end


[stable_necc, stab_local_contr, Big_Q_sym] = check_lags_LMI_constraints(x_test, alpha_fun, h_fun, ... 
                                         A_g, A_l, A_d, att_g, att_l, P_l, P_g, lyap_der, Mu, Sigma);

% Analyze Local Stability of EigenFunctions
Q_sym_fun       = @(x)quadratic_4d_to_2D(x, att_g, att_l, Big_Q_sym);
plot_lyap_fct_lags(Q_sym_fun, 0, limits_,  '$Q_{sym}$ Component', []); hold on;
plot_ds_model(fig1, ds_lags_single, att_g, limits,'low'); 
if exist('h_sample','var'); delete(h_sample);  end
[h_sample] = scatter(x_test(1),x_test(2),100,'+','g','LineWidth',2);

Q_test = Q_sym_fun(x_test)
[x_min, Q_min, H_Q] = min_quadratic_4d_to_2D(att_g, att_l,  Big_Q_sym);
sign_Hq = checkDefiniteness(H_Q)
if exist('h_sample_min','var'); delete(h_sample_min);  end
[h_sample_min] = scatter(x_min(1),x_min(2),100,'+','r','LineWidth',2);