%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Demo Code for Locally Active Globally Stable DS with 1 local region %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Clear MATLAB workspace
close all; clear all; clc;

%%%%%%% Choose to plot robot for simulation %%%%%%%
with_robot = 1;
if with_robot
    % set up a simple robot and a figure that plots it
    robot = create_simple_robot();
    fig1 = initialize_robot_figure(robot);
    title('Feasible Robot Workspace','Interpreter','LaTex')    
    % Base Offset
    base = [-1 1]';
    % Axis limits
    limits = [-2.5 0.5 -0.45 2.2];
else
    fig1 = figure('Color',[1 1 1]);
    % Axis limits
    limits = [-3.5 0.5 -0.75 1.75];
    axis(limits)    
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.25, 0.55, 0.2646 0.4358]);
    grid on
end

% Global Attractor of DS
att_g = [0 0]';
scatter(att_g(1),att_g(2),100,[0 0 0],'d'); hold on;

% Draw Reference Trajectory
data = draw_mouse_data_on_DS(fig1, limits);
Data = [];
for l=1:length(data)
    Data = [Data data{l}];
end

% Position/Velocity Trajectories
Xi_ref     = Data(1:2,:);
Xi_dot_ref = Data(3:end,:);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Step 2 (GMM FITTING): Fit GMM to Trajectory Data %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% GMM Estimation Algorithm %%%%%%%%%%%%%%%%%%%%%%
% 0: Physically-Consistent Non-Parametric (Collapsed Gibbs Sampler)
% 1: GMM-EM Model Selection via BIC
% 2: CRP-GMM (Collapsed Gibbs Sampler)
est_options = [];
est_options.type             = 1;   % GMM Estimation Alorithm Type   

% If algo 1 selected:
est_options.maxK             = 15;  % Maximum Gaussians for Type 1
est_options.fixed_K          = 1;  % Fix K and estimate with EM for Type 1

% If algo 0 or 2 selected:
est_options.samplerIter      = 100;  % Maximum Sampler Iterations
                                    % For type 0: 20-50 iter is sufficient
                                    % For type 2: >100 iter are needed
                                    
est_options.do_plots         = 0;   % Plot Estimation Statistics
est_options.sub_sample       = 2;   % Size of sub-sampling of trajectories
                                    % 1/2 for 2D datasets, >2/3 for real    
% Metric Hyper-parameters
est_options.estimate_l       = 1;   % '0/1' Estimate the lengthscale, if set to 1
est_options.l_sensitivity    = 0.5;   % lengthscale sensitivity [1-10->>100]
                                    % Default value is set to '2' as in the
                                    % paper, for very messy, close to
                                    % self-intersecting trajectories, we
                                    % recommend a higher value
est_options.length_scale     = [];  % if estimate_l=0 you can define your own
                                    % l, when setting l=0 only
                                    % directionality is taken into account

% Fit GMM to Trajectory Data
[Priors, Mu, Sigma] = fit_gmm(Xi_ref, Xi_dot_ref, est_options);

%% Generate GMM data structure for DS learning
clear ds_gmm; ds_gmm.Mu = Mu; ds_gmm.Sigma = Sigma; ds_gmm.Priors = Priors; 

%%  Visualize Gaussian Components and labels on clustered trajectories 
% Extract Cluster Labels
[~, est_labels] =  my_gmm_cluster(Xi_ref, ds_gmm.Priors, ds_gmm.Mu, ds_gmm.Sigma, 'hard', []);

% Visualize Estimated Parameters
[h_gmm]  = visualizeEstimatedGMM(Xi_ref,  ds_gmm.Priors, ds_gmm.Mu, ds_gmm.Sigma, est_labels, est_options);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%% Step 1b: Estimate local attractors and Construct Activation function %%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if exist('h_dec','var');  delete(h_dec); end
if exist('h_data','var'); delete(h_data);end
if exist('h_att','var');  delete(h_att); end
if exist('h_vel','var');  delete(h_vel); end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% ACTIVATION FUNCTION & LOCALLY LINEAR MODEL PARAMS %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute Unit Directions from Reference Trajectories
[Q, L, Mu] =  my_pca(Xi_ref);
% [Q, L] =  eig(Sigma);

% Estimate Local Attractor from Reference Trajectories
[att_l, A_inv] =  estimate_local_attractor(Data);

% Create Hyper-Plane function
w =  (Mu-att_l)/norm(Mu-att_l);
h_fun = @(x)hyper_plane(x,w,att_l);
% grad_h_fun = @(x)(repmat(w,[1 size(x,2)]));
grad_h_fun = @(x)grad_hyper_plane(x,w,h_fun);

% Gaussian Covariance scaling
gauss_opt_thres = 0.25; 

% Activation function
radius_fun       = @(x)(1 - my_exp_loc_act(1, att_g, x));

% alpha(\xi) -> Gaussian Basis Function
scale = 1; rel_eig = 0.25;
Sigma     = Q * L * Q';
Norm      = my_gaussPDF(Mu, Mu, Sigma);
activ_fun = @(x)(1 - (my_gaussPDF(x, Mu, Sigma)/Norm));
alpha_fun  = @(x)( (1-radius_fun(x)).*activ_fun(x)' + radius_fun(x));

% Search for best scale parameter --> should be more efficient
step = 0.25;
while activ_fun(att_l) > gauss_opt_thres
    scale = scale + step;
    Sigma     = Q * (scale*diag([L(1,1)*(1+rel_eig);L(1,1)*rel_eig]))* Q';
    Norm      = my_gaussPDF(Mu, Mu, Sigma);
    activ_fun = @(x)(1 - (my_gaussPDF(x, Mu, Sigma)/Norm));
    alpha_fun  = @(x)( (1-radius_fun(x)).*activ_fun(x)' + radius_fun(x));
end

%% Plot values of mixing function to see where transition occur
with_robot = 1;
if with_robot
    % set up a simple robot and a figure that plots it
    robot = create_simple_robot();
    fig1 = initialize_robot_figure(robot);
    title('Feasible Robot Workspace','Interpreter','LaTex')    
    % Base Offset
    base = [-1 1]';
    % Axis limits
    limits = [-2.5 0.5 -0.45 2.2];
else
    fig1 = figure('Color',[1 1 1]);
    % Axis limits
    limits = [-3.5 0.5 -0.75 1.75];
    axis(limits)    
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.25, 0.55, 0.2646 0.4358]);
    grid on
end

% Plot Mixing function
h_dec = plot_mixing_fct_2d(limits, alpha_fun); hold on;

% Position/Velocity Trajectories
vel_samples = 10; vel_size = 0.5; 
hold on;
[h_data, h_att, h_vel] = plot_reference_trajectories_DS(Data, att_g, vel_samples, vel_size, fig1);
text(att_g(1),att_g(2),'$\mathbf{\xi}^*_g$','Interpreter', 'LaTex','FontSize',15); hold on;

h_att = scatter(att_l(1),att_l(2),150,[0 0 0],'d','Linewidth',2); hold on;
text(att_l(1),att_l(2),'$\mathbf{\xi}^*_l$','Interpreter', 'LaTex','FontSize',15)
limits_ = limits + [-0.015 0.015 -0.015 0.015];
axis(limits_)
box on
xlabel('$\mathbf{\xi}_1$','Interpreter', 'LaTex','FontSize',15)
ylabel('$\mathbf{\xi}_2$','Interpreter', 'LaTex','FontSize',15)
title('Reference Trajectory, Global Attractor and Local Attractor','Interpreter', 'LaTex','FontSize',15)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%% Step 2: ESTIMATE CANDIDATE LYAPUNOV FUNCTION PARAMETERS %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Vxf]    = learn_wsaqf(Data, att_l);
P_g      = Vxf.P(:,:,1);
P_l      = Vxf.P(:,:,2);

% Lyapunov derivative and gradient functions
grad_lyap_fun = @(x)gradient_lyapunov(x, att_g, att_l, P_g, P_l);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%% Step 3: ESTIMATE Global and Local System Matrices     %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%% DS PARAMETER INITIALIZATION %%%%%%%%%%%%%%%%%%%
% Global Dynamics type
fg_type = 0;          % 0: Fixed Linear system Axi + b
                      % 1: Estimated Linear system Axi + b
% Local Dynamics type
fl_type = 1;  % 1: Symmetrically converging to ref. trajectory
              % 2: Symmetrically diverging from ref. trajectory
              % 3: Converging towards via-point   
% Visualization                      
mix_type = 1;     % 1: gaussian basis function around ref. traj.
                  % 2: Visualize global or local
dtype    = 1;     % For option 2, which dynamics to visualize
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
if mix_type ==  2 % For visualization purposes only
        % === ONLY ONE TYPE OF DYNAMICS ARE ACTIVATED ===
        if exist('h_dec','var');  delete(h_dec); end
        alpha_fun = @(x)(dtype*ones(1,size(x,2)));                    
end

% Estimate Global Dynamics from Data
if fg_type == 0
    A_g = -2*eye(2); b_g = -A_g*att_g;
else
    [A_g, b_g, ~] = optimize_linear_ds_from_data(Data, att_g, fg_type, 1, P_g, Mu, P_l, att_l, 'full');       
end

% Stability Checks
Q_g  = A_g'*P_g + P_g*A_g
lambda_Qg = eig(Q_g)
Q_gl = A_g'*P_l' 
lambda_Qgl = eig(Q_gl)

% Local Attractor diffusive function component
breadth_mod = 15;
lambda_fun = @(x)lambda_mod_fun(x, breadth_mod, att_l, grad_h_fun, grad_lyap_fun);

%% Option 1: Estimate Local Dynamics from Data by minimizing velocity error with given tracking/known factor
kappa = 500
[A_l, b_l, A_d, b_d] = estimate_localDS_known_gamma(Data, A_g,  att_g, att_l, fl_type, kappa,w, P_g, P_l, Q);
Lambda_l = eig(A_l)
kappa = max(abs(Lambda_l))/min(abs(Lambda_l))

%% Option 2: Estimate Local Dynamics by max. tracking factor + min. velocity mse
% Construct variables and function handles for stability constraints
clear stability_vars 
stability_vars.solver          = 'fmincon'; % options: 'baron' or 'fmincon'
stability_vars.grad_h_fun      = grad_h_fun;
stability_vars.add_constr      = 1; % 0: no stability constraints
                                    % 1: Adding sampled stability constraints

if stability_vars.add_constr   
    
    % Draw Initial set of samples for point-wise stability constraints
    desired_samples = 100;
    chi_samples = draw_chi_samples (Sigma,Mu,desired_samples,activ_fun);
        
    % Function handles for contraint evaluation
    stability_vars.chi_samples   = chi_samples;
    stability_vars.alpha_fun     = alpha_fun;
    stability_vars.h_fun         = h_fun;
    stability_vars.lambda_fun    = lambda_fun;
    
    
    % Type of constraint to evaluate
    stability_vars.constraint_type = 'hessian';    % options: 'full/matrix/hessian'
    
    % Variable for each type
    if strcmp(stability_vars.constraint_type,'full')
        stability_vars.grad_lyap_fun = grad_lyap_fun;
    else 
        stability_vars.P_l           = P_l;
        stability_vars.P_g           = P_g;
    end
                
    % Local DS Optimization with Constraint Rejection Sampling
    num_violations = 1; iter = 1;
    while num_violations > 0
        fprintf('Iteration %d, using %d chi-samples..\n', iter, size(chi_samples,2));
        stability_vars.chi_samples   = chi_samples;
 
        if strcmp(stability_vars.constraint_type,'hessian')
            [A_l, b_l, A_d, b_d] = optimize_localDS_for_LAGS_Hess(Data, A_g, att_g, stability_vars);
        else
            [A_l, b_l, A_d, b_d] = optimize_localDS_for_LAGS(Data, A_g, att_g, fl_type, stability_vars);
        end
        
        %%%% Check for violations in compact set %%%%        
        fprintf(2, 'Checking violation of Lyapunov Constraint.. ');
        clear lyap_der
        lyap_der = @(x)lyapunov_combined_derivative_full(x, att_g, att_l, P_g, P_l, alpha_fun, h_fun, lambda_fun, grad_h_fun, A_g, A_l, A_d);
        
        % Samples to evaluate
        desired_test_samples = 50000;
        chi_samples_test = draw_chi_samples (Sigma,Mu,desired_test_samples,activ_fun);
        
        % Evaluate Samples
        necc_lyap_constr = lyap_der(chi_samples_test);
        
        % Necessary Constraints
        necc_violations = necc_lyap_constr >= 0;
        necc_violating_points = chi_samples_test(:,necc_violations);
        num_violations = size(necc_violating_points,2);                                
        fprintf(2, '. done. \n');
        
        % If violations sample new contraint-eval points
        if num_violations > 0
            fprintf(2, 'Violations for Lyapunov Constraint %d..\n', num_violations);
            if num_violations < 10
                chi_samples = [chi_samples necc_violating_points];
            else
                num_new_samples = ceil(num_violations/5);
                new_chi_samples = necc_violating_points(:,randsample(num_violations,num_new_samples));
                chi_samples = [chi_samples new_chi_samples];
            end
        else
            fprintf('Optimization converged...\n');
        end                
        iter = iter + 1;
    end
else
    tic;
    [A_l, b_l, A_d, b_d] = optimize_localDS_for_LAGS(Data, A_g, att_g, fl_type, stability_vars);
    toc;
end

Lambda_l = eig(A_l)
kappa = max(abs(Lambda_l))/min(abs(Lambda_l))

%% Post-learning Numerical Stability Check using full Lyapunov Derivative
% Function for Lyap-Der evaluation
lyap_der = @(x)lyapunov_combined_derivative_full(x, att_g, att_l, P_g, P_l, alpha_fun, h_fun, lambda_fun, grad_h_fun, A_g, A_l, A_d);

% Samples to evaluate
desired_samples = 50000;
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
eig(H_Q)
trace(H_Q)
if exist('h_sample_min','var'); delete(h_sample_min);  end
[h_sample_min] = scatter(x_min(1),x_min(2),100,'+','r','LineWidth',2);

%% %%%%%%%%%%%%    Generate/Plot Resulting DS  %%%%%%%%%%%%%%%%%%%
% Function for DS
ds_lags_single = @(x) lags_ds(att_g, x, mix_type, alpha_fun, A_g, b_g, A_l, b_l, att_l, h_fun, A_d, b_d, lambda_fun, grad_h_fun);

% Plot Dynamical System
if exist('hs','var');     delete(hs);    end
hs = plot_ds_model(fig1, ds_lags_single, att_g, limits,'high'); hold on;
if (mix_type ==  2)
    if (dtype ==  1)
        title('Global DS $\dot{\xi}= f_g(\xi) = A_g\xi + b_g$ ', 'Interpreter','LaTex','FontSize',15)
    else
        title('Local DS $\dot{\xi} = f_l(\xi) = A_l(h(\xi))(\xi - \xi_l^*) - \lambda(\xi)\nabla_{\xi}h(\xi)$ ', 'Interpreter','LaTex','FontSize',18);
    end
else
    title('Locally Active - Globally Stable (LAGS) $\dot{\xi}=\alpha(\xi)f_g(\xi) + (1 - \alpha(\xi))f_l(\xi)$', 'Interpreter','LaTex','FontSize',18)
end
box on

%% Simulate Passive DS Controller function
if with_robot
    dt = 0.005;
    simulate_passiveDS(fig1, robot, base, ds_lags_single, att_g, dt);
end

%% Plot WSAQF Lyapunov Function and derivative -- NEW
% Symmetric of Asymmetric QF's
lyap_type = 1; % 0: WSQF, 1: WSAQF

% Type of plot
contour = 1; % 0: surf, 1: contour
clear lyap_fun_comb lyap_der 

switch lyap_type
    case 0        
        % Lyapunov function
        lyap_fun_comb = @(x)lyapunov_function_combined(x, att_g, att_l, lyap_type);        
        % Derivative of Lyapunov function
        lyap_der = @(x)lyapunov_combined_derivative(x, att_g, att_l, ds_lags_single, lyap_type);
        title_string = {'WSQF Candidate $V(\xi) = (\xi -\xi_g^*)^T(\xi-\xi_g^*) + \beta((\xi-\xi_g^*)^T(\xi-\xi_l^*))^2$'};
    case 1     
        
        % Lyapunov function
        lyap_fun_comb = @(x)lyapunov_function_combined(x, att_g, att_l, lyap_type, P_g, P_l);        
        
        % Derivative of Lyapunov function (gradV*f(x))
%         lyap_der = @(x)lyapunov_combined_derivative(x, att_g, att_l, ds_lags_single, lyap_type, P_g, P_l, beta_eps);        
        
        % Derivative of Lyapunov function (fully factorized)
        lyap_der = @(x)lyapunov_combined_derivative_full(x, att_g, att_l, P_g, P_l, alpha_fun, h_fun, lambda_fun, grad_h_fun, A_g, A_l, A_d);
        title_string = {'$V(\xi) = (\xi-\xi_g^*)^TP_g(\xi-\xi_g^*) + \beta((\xi-\xi_g^*)^TP_l(\xi-\xi_l^*))^2$'};
end
title_string_der = {'Lyapunov Function Derivative $\dot{V}(\xi)$'};

% if exist('h_lyap','var');     delete(h_lyap);     end
% if exist('h_lyap_der','var'); delete(h_lyap_der); end
h_lyap     = plot_lyap_fct(lyap_fun_comb, contour, limits_,  title_string, 0);
h_lyap_der = plot_lyap_fct(lyap_der, contour, limits_,  title_string_der, 1);

%% Compare Velocities from Demonstration vs DS
% Simulated velocities of DS converging to target from starting point
xd_dot = []; xd = [];
% Simulate velocities from same reference trajectory
for i=1:length(Data)
    xd_dot_ = ds_lags_single(Data(1:2,i));    
    % Record Trajectories
    xd_dot = [xd_dot xd_dot_];        
end

% Plot Demonstrated Velocities vs Generated Velocities
if exist('h_vel','var');     delete(h_vel);    end
h_vel = figure('Color',[1 1 1]);
plot(Data(3,:)', '.-','Color',[0 0 1], 'LineWidth',2); hold on;
plot(Data(4,:)', '.-','Color',[1 0 0], 'LineWidth',2); hold on;
plot(xd_dot(1,:)','--','Color',[0 0 1], 'LineWidth', 1); hold on;
plot(xd_dot(2,:)','--','Color',[1 0 0], 'LineWidth', 1); hold on;
grid on;
legend({'$\dot{\xi}^{ref}_{x}$','$\dot{\xi}^{ref}_{y}$','$\dot{\xi}^{d}_{x}$','$\dot{\xi}^{d}_{y}$'}, 'Interpreter', 'LaTex', 'FontSize', 15)

