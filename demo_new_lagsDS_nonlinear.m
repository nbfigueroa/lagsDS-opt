%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demo Script for GMM-based LAGS-DS Learning introduced in paper:         %
%  'Locally Active Globally Stable Dynamical Systems';                    %
% N. Figueroa and A. Billard; TRO/IJRR 2019                               %
% With this script you can load 2D toy trajectories and real-world        %
% trajectories acquired via kinesthetic taching and test the different    %                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2019 Learning Algorithms and Systems Laboratory,          %
% EPFL, Switzerland                                                       %
% Author:  Nadia Figueroa                                                 % 
% email:   nadia.figueroafernandez@epfl.ch                                %
% website: http://lasa.epfl.ch                                            %
%                                                                         %
% This work was supported by the EU project Cogimon H2020-ICT-23-2014.    %
%                                                                         %
% Permission is granted to copy, distribute, and/or modify this program   %
% under the terms of the GNU General Public License, version 2 or any     %
% later version published by the Free Software Foundation.                %
%                                                                         %
% This program is distributed in the hope that it will be useful, but     %
% WITHOUT ANY WARRANTY; without even the implied warranty of              %
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General%
% Public License for more details                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Demo Code for Locally Active Globally Stable DS with multiple local regions %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  DATA LOADING OPTION 1: Draw with GUI %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clear all; clc
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
%     limits = [-2.5 0.5 -0.45 2.2];
%     limits = [-2.5 0.5 -0.45 1.5];
    limits = [-2.5 0.5 -0.45 2.2];
else
    fig1 = figure('Color',[1 1 1]);
    % Axis limits
%     limits = [-3.5 0.5 -0.5 2];
%     limits = [-2.5 0.5 -0.45 1.5];
%     limits = [-0.5 2.5 -1.5 0.45];
%     limits = [-6 4 -2 2];
    limits = [-2.5 0.5 -0.45 2.2];
    axis(limits)    
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.25, 0.55, 0.2646 0.4358]);
    grid on
end

% Global Attractor of DS
att_g = [0 0]';
scatter(att_g(1),att_g(2),100,[0 0 0],'d'); hold on;

% Draw Reference Trajectory
data = draw_mouse_data_on_DS(fig1, limits);
% Choose Activation Function type
B_r = 10;
radius_fun = @(x)(1 - my_exp_loc_act(B_r, att_g, x));
grad_radius_fun = @(x)grad_lambda_fun(x, B_r, att_g);

Data = [];
for l=1:length(data)    
    % Check where demos end and shift
    data_ = data{l};
    if radius_fun(data_(1:2,end)) > 0.75
        data_(1:2,:) = data_(1:2,:) - repmat(data_(1:2,end), [1 length(data_)]);
        data_(3:4,end) = zeros(2,1);
    end    
    Data = [Data data_];
end

% Position/Velocity Trajectories
Xi_ref     = Data(1:2,:);
Xi_dot_ref = Data(3:end,:);

%%  DATA LOADING OPTION 2: From LASA DATASET
% Global Attractor of DS
att_g = [0 0]';

% Choose Activation Function type
B_r = 10;
radius_fun = @(x)(1 - my_exp_loc_act(B_r, att_g, x));
grad_radius_fun = @(x)grad_lambda_fun(x, B_r, att_g);

% Choose DS LASA Dataset to load
[demos, limits] = demo_load_LASA_dataset();

sample = 1;
Data = [];
for l=1:2   
    % Check where demos end and shift    
    data_pos_raw = demos{l}.pos(:,1:sample:end); 
    data_filt    = sgolay_time_derivatives(data_pos_raw', demos{l}.dt, 2, 3, 15);
    data_ = [data_filt(:,:,1), data_filt(:,:,2)]';
    if radius_fun(data_(1:2,end)) > 0.75
        data_(1:2,:) = data_(1:2,:) - repmat(data_(1:2,end), [1 length(data_)]);
        data_(3:4,end) = zeros(2,1);
    end    
    Data = [Data data_];
    clear data_
end

% Position/Velocity Trajectories
Xi_ref     = Data(1:2,:);
Xi_dot_ref = Data(3:end,:);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%% Step 1a: Discover Local Models with Selected GMM Estimation type %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% GMM Estimation Algorithm %%%%
% 0: Physically-Consistent Non-Parametric
% 1: GMM-EM Model Selection via BIC
% 2: GMM via Competitive-EM
est_options = [];
est_options.type       = 0;   % GMM Estimation Alorithm Type    
est_options.maxK       = 10;  % Maximum Gaussians for Type 1/2
est_options.do_plots   = 1;   % Plot Estimation Statistics
est_options.adjusts_C   = 1;   % Adjust Sigmas
est_options.fixed_K     = [];  % Fix K and estimate with EM
est_options.exp_scaling = 1;   % Scaling for the similarity to improve locality

% Discover Local Models
sample = 2;
[Priors0, Mu0, Sigma0] = discover_local_models(Xi_ref(:,1:sample:end), Xi_dot_ref(:,1:sample:end), est_options);

%% Extract Cluster Labels
est_K      = length(Priors0); 
Priors = Priors0; Mu = Mu0; Sigma = Sigma0;
[~, est_labels] =  my_gmm_cluster(Xi_ref, Priors, Mu, Sigma, 'hard', []);

% Visualize Cluster Parameters on Manifold Data
plotGMMParameters( Xi_ref, est_labels, Mu, Sigma);
limits_ = limits + [-0.015 0.015 -0.015 0.015];
axis(limits_)
switch est_options.type   
    case 0
        title('Physically-Consistent Non-Parametric Mixture Model','Interpreter','LaTex', 'FontSize',15);
    case 1        
        title('Best fit GMM with EM-based BIC Model Selection','Interpreter','LaTex', 'FontSize',15);
    case 3
        title('Bayesian Non-Parametric Mixture Model','Interpreter','LaTex', 'FontSize',15);
end

%%% Visualize GMM pdf from learnt parameters (for 1D Datasets)
ml_plot_gmm_pdf(Xi_ref, Priors, Mu, Sigma, limits)
clear ds_gmm; ds_gmm.Mu = Mu; ds_gmm.Sigma = Sigma; 
ds_gmm.Priors = Priors; 

% Adjust Covariance Matrices
est_options.adjusts_C  = 1;
if est_options.adjusts_C  == 1
    tot_scale_fact = 1; rel_scale_fact = 0.15;
    Sigma = adjust_Covariances(Sigma0, tot_scale_fact, rel_scale_fact);
    ds_gmm.Sigma = Sigma;
    % Visualize Cluster Parameters on Manifold Data
    plotGMMParameters( Xi_ref, est_labels, Mu, Sigma);
    limits_ = limits + [-0.015 0.015 -0.015 0.015];
    axis(limits_)
    switch est_options.type   
    case 0
        title('Physically-Consistent Non-Parametric Mixture Model','Interpreter','LaTex', 'FontSize',15);
    case 1        
        title('Best fit GMM with EM-based BIC Model Selection','Interpreter','LaTex', 'FontSize',15);
    case 3
        title('Bayesian Non-Parametric Mixture Model','Interpreter','LaTex', 'FontSize',15);
    end   
    
    ml_plot_gmm_pdf(Xi_ref, Priors, Mu, Sigma, limits)  

end    
   

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%% Step 1b: Estimate local attractors and Construct Activation function %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Construct GMM for DS mixing functions
if exist('hs','var');     delete(hs);    end
if exist('h_data','var'); delete(h_data);end
if exist('h_att','var');  delete(h_att); end
if exist('h_vel','var');  delete(h_vel); end

% Compute Hyper-plane functions per local model
w = zeros(2,est_K); breadth_mod = 50;
h_functor = cell(1,est_K); grad_h_functor = cell(1,est_K);
lambda_functor  = cell(1,est_K); Data_k = [];
for k=1:est_K
    k
    % Estimate Local Attractors from Reference Trajectories and GMM
    Data_k{k}  = Data(:,est_labels==k)
    att_l(:,k) = Data_k{k}(1:2,end)        
    
    % TODO: FIX THIS!
    [att_l(:,k), ~] =  estimate_local_attractor(Data_k{k});
    if radius_fun(att_l(:,k)) > 0.5
        att_l(:,k) = att_g;
    end  
    
    % Create Hyper-Plane functions
    w(:,k) =  (Mu(:,k)-att_l(:,k))/norm(Mu(:,k)-att_l(:,k));
    h_functor{k} = @(x)hyper_plane(x,w(:,k),att_l(:,k));
    grad_h_functor{k} = @(x)grad_hyper_plane(x,w(:,k),h_functor{k});
    lambda_functor{k} = @(x)(1-my_exp_loc_act(breadth_mod, att_l(:,k), x));
    grad_lambda_functor{k} = @(x)grad_lambda_fun(x, breadth_mod, att_l(:,k));
end

%% === ACTIVATION USING GMM Directly ===
if exist('h_dec','var');  delete(h_dec); end
act_type = 0;
switch act_type
    case 0
        % Option 1: with normalized pdf of gmm
        % GMM-PDF
        gmm_fun    = @(x)ml_gmm_pdf(x, ds_gmm.Priors, ds_gmm.Mu, ds_gmm.Sigma);
        activ_fun  = @(x) 1 - gmm_fun(x);
        
        % Direct vs Max. Alpha functions
        alpha_fun  = @(x)( (1-radius_fun(x))'.*max(0,activ_fun(x))' + radius_fun(x)');
        
        % Compute gradient       
        grad_gmm_fun   = @(x)grad_gmm_pdf(x, ds_gmm);
        grad_alpha_fun = @(x)gradient_alpha_fun(x,radius_fun, grad_radius_fun, gmm_fun, grad_gmm_fun, 'gmm');

    case 1 % With GPR
        % Train GPR model of the scaling data \Xi,\Kappa
        Data_ = Data(1:2,:);
        % If we want to choose a region
        local_k = [2 3]; choose_k = 0;
        if choose_k
            Data_ = [];
            for k_=1:length(local_k)
            Data_ = [Data_ Data(1:2,est_labels_init==local_k(k_))];
            end
        end
        X_train = Data_; y_train = ones(1,length(X_train));        
        epsilon = 0.001; rbf_var = 0.1; % assuming output variance = 1
        model.X_train   = X_train';   model.y_train   = y_train';
        gpr_fun    = @(x) my_gpr(x',[],model,epsilon,rbf_var);
        activ_fun  = @(x) 1 - gpr_fun(x);                    
        alpha_fun  = @(x)( (1-radius_fun(x))'.*activ_fun(x)' + radius_fun(x)');        
        
        % Compute gradient       
        grad_gpr_fun   = @(x)gradient_gpr(x, model, epsilon, rbf_var);
        grad_alpha_fun = @(x)gradient_alpha_fun(x,radius_fun, grad_radius_fun, gpr_fun, grad_gpr_fun, 'gpr');        
end

% Plot values of mixing function to see where transition occur
h_dec = plot_mixing_fct_2d(limits, alpha_fun); hold on;

% Plot Reference Trajectories on Top
plot_attr = 1;
if plot_attr
    if exist('h_att_g','var');  delete(h_att_g); end
    if exist('h_att_l','var');  delete(h_att_l); end
    [h_data, h_att, h_vel] = plot_reference_trajectories(Data, att_g, att_l, 5);
    h_att_g = text(att_g(1),att_g(2),'$\mathbf{\xi}^*_g$','Interpreter', 'LaTex','FontSize',20);
    h_att_l = [];
    for k=1:est_K
        att_string = sprintf('$$\\xi^*_%i$$',k);
        h_att_l = [h_att_l text(att_l(1,k),att_l(2,k),att_string,'Interpreter', 'LaTex','FontSize',20)];
    end
    limits_ = limits + [-0.015 0.015 -0.015 0.015];
    axis(limits_)
    box on
end
xlabel('$\xi_1$','Interpreter', 'LaTex','FontSize',15)
ylabel('$\xi_2$','Interpreter', 'LaTex','FontSize',15)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%  Step 2: ESTIMATE CANDIDATE LYAPUNOV FUNCTION PARAMETERS  %%%%%%%%%
%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Vxf]    = learn_wsaqf(Data, att_l);
P_g      = Vxf.P(:,:,1)
P_l      = Vxf.P(:,:,2:end)

% If any local attractor is equivalent to the global re-parametrize
equal_g = sum(att_l == zeros(2,1));
if any(equal_g)
    id_l = find(equal_g>0);
    P_g = P_g + P_l(:,:,id_l)
    P_l(:,:,id_l) = P_g;    
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%  Step 3: ESTIMATE SYSTEM DYNAMICS MATRICES  %%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% DS PARAMETER INITIALIZATION %%%%%%%%%%%%%%%%%%%
% Global Dynamics type
Ag_type  = 1;       % 0: Fixed Linear system Axi + b
                    % 1: LPV system sum_{k=1}^{K}h_k(xi)(A_kxi + b_k)`
                    
% Type of constraints to estimate Axi+b or LPV
constr_type = 3;      % 0:'convex' with QLF:     A' + A < 0
                      % 1:'non-convex' with P-QLF: A'P + PA < -Q
                      % 2:'non-convex' with P-QLF: A'P + PA < -Q given P
                      % 3:'non-convex' with WSAQF
                      
% Local Dynamics type (dependent on K local linear models)
ds_types = 1*ones(1,est_K);
                    % 1: Symmetrically converging to ref. trajectory
                    % 2: Symmetrically diverging from ref. trajectory
                    % 3: Converging towards 'shifted' equilibrium 

% Choose activation function for local DS's
act_type   = 0;   % 0: No local dynamics`
                  % 1: With local dynamics               
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Over-ride alpha function to visualize global or local DS
if act_type == 0
    % === ONLY GLOBAL DYNAMICS ARE ACTIVATED ===
    if exist('h_dec','var');  delete(h_dec); end
    alpha_fun     = @(x)(1*ones(1,size(x,2)));
    grad_alpha_fun = @(x)(zeros(2,size(x,2)));
    %     plotGMMParameters( Xi_ref, est_labels, Mu, Sigma, fig1);
end   

% Fit/Choose A_g for demonstrated Data
switch Ag_type
    case 0 % 0: Fixed Linear system Axi + b
        A_g = [-5 0; 0 -5]; b_g = -A_g*zeros(2,1);
        
    case 1 % 1: LPV system sum_{k=1}^{K}h_k(xi)(A_kxi + b_k)            
        
        if constr_type == 3
            [A_g, b_g, P_est] = optimize_lpv_ds_from_data_v2(Data, att_g, constr_type, ds_gmm, P_g, P_l);
        else
             [A_g, b_g, P_est] = optimize_lpv_ds_from_data_v2(Data, att_g, constr_type, ds_gmm, P_g);
        end
        
        %%%% FOR DEBUGGING: Check Negative-Definite Constraint %%%%
        constr_violations = zeros(1,est_K);
        suff_constr_violations = zeros(1,est_K);
        for k=1:est_K
            A_t = A_g(:,:,k) + A_g(:,:,k)';
            constr_violations(1,k) = sum(eig(A_t) > 0); % strict
            Pg_A = (P_g' + P_g) * A_g(:,:,k);
            suff_constr_violations(1,k) = sum(eig(Pg_A + Pg_A') > 0); % strict
        end
        
        % Check Constraint Violation
        if sum(constr_violations) > 0
            warning(sprintf('Strict System Matrix Constraints are NOT met..'))
        else
            fprintf('All Strict System Matrix Constraints are met..\n')
        end
        
        % Check Constraint Violation
        if sum(suff_constr_violations) > 0
            warning(sprintf('Sufficient System Matrix Constraints are NOT met..'))
        else
            fprintf('All Sufficient System Matrix Constraints are met..\n')
        end        
end

% -- TODO: This can be done inside the estmation function
% Check full constraints along reference trajectory
x_test = Xi_ref;
full_constr_viol = zeros(1,size(x_test,2));
gamma_k_x = posterior_probs_gmm(x_test,ds_gmm,'norm');
for i=1:size(x_test,2)
     A_g_k = zeros(2,2); P_l_k = zeros(2,2);
     for k=1:est_K
         % Compute weighted A's
         A_g_k = A_g_k + gamma_k_x(k,i) * A_g(:,:,k);
         
         % Compute weighted P's
         lyap_local_k =   (x_test(:,i) - att_g)'*P_l(:,:,k)*(x_test(:,i) - att_l(:,k));
         
         % Computing activation term
         if lyap_local_k >= 0
             beta = 1;
         else
             beta = 0;
         end
         beta_k_2 = 2 * beta * lyap_local_k;
         P_l_k = P_l_k + beta_k_2*P_l(:,:,k);         
     end     
     % Compute Q_K
     AQ = A_g_k'*(2*P_g  + P_l_k);
     full_constr_viol(1,i) = sum(eig(AQ+AQ') > 0);
end
% Check Constraint Violation
if sum(full_constr_viol) > 0
    warning(sprintf('Full System Matrix Constraints are NOT met..'))
else
    fprintf('Full System Matrix Constraints are met..\n')
end

%% Option 1: Estimate Local Dynamics from Data by minimizing velocity error with given tracking factor
ds_types = [3 3 3 3 3 3 3]
tracking_factor = [5 15 10 10 10 5 10]; % Relative ratio between eigenvalues
A_l = []; b_l = []; A_d = []; b_d = [];
for k=1:est_K
    % Compute lamda's from reference trajectories
    if Ag_type == 1
        [A_l(:,:,k), ~, A_d(:,:,k), b_d(:,k)] = estimate_localDS_known_gamma(Data_k{k}, A_g(:,:,k),  att_g, att_l(:,k), ds_types(k), tracking_factor(k),w(:,k));
    else
        [A_l(:,:,k), ~, A_d(:,:,k), b_d(:,k)] = estimate_localDS_known_gamma(Data_k{k}, A_g, att_g, att_l(:,k), ds_types(k), tracking_factor(k),w(:,k));
    end 
    
    if any(att_l(:,k)==att_g)
        A_d(:,:,k) = -eye(2);
        b_d(:,k) = zeros(2,1);
    end
    
    b_l(:,k) = -A_l(:,:,k)*att_l(:,k);    
    Lambda_l = eig(A_l(:,:,k));
    gamma = max(abs(Lambda_l))/min(abs(Lambda_l))
end

%% Option 2: Estimate Local Dynamics by max. tracking factor + min. reconstruction error with Lyapunov Stability Constraints!
% Draw samples for point-wise stability constraints
desired_samples = 100;
chi_samples = draw_chi_samples (Sigma,Mu,desired_samples,activ_fun);

% Construct variables and function handles for stability constraints
grad_lyap_fun = @(x)gradient_lyapunov(x, att_g, att_l, P_g, P_l);
clear stability_vars 
stability_vars.add_constr    = 1;
stability_vars.chi_samples   = chi_samples;
stability_vars.grad_lyap_fun = grad_lyap_fun;
stability_vars.alpha_fun     = alpha_fun;
stability_vars.h_fun         = h_fun; 
stability_vars.lambda_fun    = lambda_fun; 
stability_vars.grad_h_fun    = grad_h_fun;

[A_l, b_l, A_d, b_d] = optimize_localDS_for_LAGS(Data, A_g, att_g, fl_type, w, stability_vars);
Lambda_l = eig(A_l);
gamma = max(abs(Lambda_l))/min(abs(Lambda_l))

%% %%%%%%%%%%%%    Plot/Generate Resulting DS  %%%%%%%%%%%%%%%%%%%
% Create Generalized Multi-Linear Compliant DS
modulation = 1;
ds_lags_multi = @(x) lags_ds_nonlinear(x, alpha_fun, ds_gmm, A_g, b_g, A_l, b_l, A_d, b_d, h_functor, lambda_functor, grad_h_functor, att_g, att_l, modulation);

if exist('hs','var');     delete(hs);    end
[hs] = plot_ds_model(fig1, ds_lags_multi, att_g, limits,'medium'); hold on;
xlabel('$x_1$','Interpreter','LaTex','FontSize',20);
ylabel('$x_2$','Interpreter','LaTex','FontSize',20);
limits_ = limits + [-0.015 0.015 -0.015 0.015];
axis(limits_)
box on
grid on
if act_type == 1  
    title('LAGS-DS $\dot{\xi}=\alpha(\xi)f_g(\xi) + (1 - \alpha(\xi))f_l(\xi)$', 'Interpreter','LaTex')
else
    title('Global-DS LPV-OPT $\dot{\xi}=\sum_{k=1}^K\gamma^k(\xi)f_g^k(\xi)$', 'Interpreter','LaTex')
end

%% Simulate Passive DS Controller function
struct_stiff = [];
struct_stiff.DS_type             = 'lags'; % Options: SEDS, Global-LPV, LAGS, LMDS ?
struct_stiff.gmm                 = ds_gmm;
struct_stiff.A_g                 = A_g;
struct_stiff.A_l                 = A_l;
struct_stiff.A_d                 = A_d;
struct_stiff.att_k               = att_l;
struct_stiff.basis               = 'D'; % Options: D (using the same basis as D) or I (using the world basis)
struct_stiff.alpha_fun           = alpha_fun;
struct_stiff.grad_alpha_fun      = grad_alpha_fun;
struct_stiff.h_functor           = h_functor;
struct_stiff.grad_h_functor      = grad_h_functor;
struct_stiff.lambda_functor      = lambda_functor;
struct_stiff.grad_lambda_functor = grad_lambda_functor;

if with_robot
    dt = 0.01;
    % Select this one if we want to see the stiffness matrices
%     simulate_passiveDS(fig1, robot, base, ds_lags_multi, att_g, dt, struct_stiff);
    simulate_passiveDS(fig1, robot, base, ds_lags_multi, att_g, dt);
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot WSAQF Lyapunov Function and derivative -- NEW
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Type of plot
contour = 1; % 0: surf, 1: contour
clear lyap_fun_comb lyap_der 

% Lyapunov function
lyap_fun_comb = @(x)lyapunov_function_combined(x, att_g, att_l, 1, P_g, P_l, ds_gmm);
title_string = {'$V_{DD-WSAQF}(\xi) = (\xi-\xi_g^*)^TP_g(\xi-\xi_g^*) + \sum_{k=1}^K\beta^k((\xi-\xi_g^*)^TP_l^k(\xi-\xi_k^*))^2$'};

% Derivative of Lyapunov function (gradV*f(x))
lyap_der = @(x)lyapunov_combined_derivative(x, att_g, att_l, ds_lags_multi, 1, P_g, P_l);
title_string_der = {'Lyapunov Function Derivative $\dot{V}_{DD-WSAQF}(\xi)$'};

% if exist('h_lyap','var');     delete(h_lyap);     end
% if exist('h_lyap_der','var'); delete(h_lyap_der); end
h_lyap     = plot_lyap_fct(lyap_fun_comb, contour, limits,  title_string, 0);
h_lyap_der = plot_lyap_fct(lyap_der, contour, limits_,  title_string_der, 1);

%% Compare Velocities from Demonstration vs DS
% Simulated velocities of DS converging to target from starting point
xd_dot = []; xd = [];
% Simulate velocities from same reference trajectory
for i=1:length(Data)
    xd_dot_ = ds_lags_multi(Data(1:2,i));    
    % Record Trajectories
    xd_dot = [xd_dot xd_dot_];        
end

% Plot Demonstrated Velocities vs Generated Velocities
if exist('h_vel','var');     delete(h_vel);    end
h_vel = figure('Color',[1 1 1]);
plot(Data(3,:)', '.-','Color', [0 0 1], 'LineWidth',2); hold on;
plot(Data(4,:)', '.-','Color', [1 0 0], 'LineWidth',2); hold on;
plot(xd_dot(1,:)','--','Color',[0 0 1], 'LineWidth', 1); hold on;
plot(xd_dot(2,:)','--','Color',[1 0 0], 'LineWidth', 1); hold on;
grid on;
legend({'$\dot{\xi}^{ref}_{x}$','$\dot{\xi}^{ref}_{y}$','$\dot{\xi}^{d}_{x}$','$\dot{\xi}^{d}_{y}$'}, 'Interpreter', 'LaTex', 'FontSize', 15)

%% Tests for mixing and activation function gradients
contour = 1;

% Compute gradient
grad_gmm_fun   = @(x)grad_gmm_pdf(x, ds_gmm);
grad_alpha_fun = @(x)gradient_alpha_fun(x,radius_fun, grad_radius_fun, gmm_fun, grad_gmm_fun, 'gmm');

% Functions to evaluate
eval_fun       = @(x)alpha_fun(x);
grad_eval_fun  = @(x)grad_alpha_fun(x);

h_gamma      = plot_lyap_fct(eval_fun, contour, limits,  {'$act$ Function'}, 1);
h_grad_gamma = plot_gradient_fct(grad_eval_fun, limits,  '$r$ and $\nabla_{\xi}r$ Function');


