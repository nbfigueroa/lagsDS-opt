%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demo Script for GMM-based LAGS-DS Learning introduced in paper:         %
%  'Locally Active Globally Stable Dynamical Systems';                    %
% N. Figueroa and A. Billard; TRO/IJRR 2019                               %
% With this script you can load/draw 2D toy trajectories and real-world   %
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Demo Code for Non-linear LAGS-DS with multiple local regions %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Step 1 [OPTION 1]: Load 2D Data Drawn from GUI %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clear all; clc
%%%%%%% Choose to plot robot for simulation %%%%%%%
with_robot   = 1;
continuous   = 1;

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
    limits = [-2.5 0.5 -0.45 2.2];
    axis(limits)    
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.25, 0.55, 0.2646 0.4358]);
    grid on
end

% Draw Reference Trajectory
data = draw_mouse_data_on_DS(fig1, limits);
if continuous
    % Process Drawn Data for DS learning
    [Data, Data_sh, att_g, x0_all, dt] = processDrawnData(data);
else
    % Process dis-joint data another way...    
    att_g = [0;0];    
end
h_att = scatter(att_g(1),att_g(2), 150, [0 0 0],'d','Linewidth',2); hold on;

% Position/Velocity Trajectories
M          = length(Data);
Xi_ref     = Data(1:2,:);
Xi_dot_ref = Data(3:end,:);
N = 2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Step 1 [OPTION 2]: Load 2D Data from Real Locomotion Datasets %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; clc; close all;
pkg_dir = '/home/nbfigueroa/Dropbox/PhD_papers/LAGS-paper/new-code/lagsDS-opt/';
%%%%%%%%%%%%%%%%%%% Choose a Dataset %%%%%%%%%%%%%%%%%%%%%                     
choosen_dataset = 3; % 1: Demos from Gazebo Simulations (right trajectories)
                     % 2: Demos from Gazebo Simulations (left+right trajectories)
                     % 3: Demos from Real iCub 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sub_sample      = 20; % To sub-sample trajectories   
[Data, Data_sh, att_g, x0_all, dt, data, ~, ...
    ~, ~, dataset_name, box_size] = load_loco_datasets(pkg_dir, choosen_dataset, sub_sample);

%% Plot Position/Velocity Trajectories %%%%%
vel_samples = 20; vel_size = 0.5; 
[h_data, h_att, h_vel] = plot_reference_trajectories_DS(Data_mean, att_g, vel_samples, vel_size);
axis equal;
limits = axis;
h_att = scatter(att_g(1),att_g(2), 150, [0 0 0],'d','Linewidth',2); hold on;

% Position/Velocity Trajectories
[~,M]      = size(Data);
Xi_ref     = Data(1:2,:);
Xi_dot_ref = Data(3:end,:);
N = 2;

%% Compute the Mean trajectory for the Locally Active Region Estimation
sub_sample      = 1;
[Data_mean] = getMeanTrajectory(data, [0;0], sub_sample, dt);

%% For 2D-locomotion Datasets
switch choosen_dataset
    case 1
        %%%%% Draw Obstacle %%%%%
        rectangle('Position',[-1 1 6 1], 'FaceColor',[.85 .85 .85 0.5]); hold on;
    case 2
        %%%%% Draw Obstacle %%%%%
        rectangle('Position',[-1 1 6 1], 'FaceColor',[.85 .85 .85 0.5]); hold on;
    case 3
        %%%%% Draw Table %%%%%
        rectangle('Position',[-6.75 -2.15 0.5 0.5], 'FaceColor',[.85 .85 .85 0.5]); hold on;
end
title_name = strcat('Reference Trajectories from:', dataset_name);
title(title_name,'Interpreter','LaTex','FontSize',16);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Step 2 (Locally Linear State-Space Paritioning): Fit GMM to Trajectory Data  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% GMM Estimation Algorithm %%%%%%%%%%%%%%%%%%%%%%
% 0: Physically-Consistent Non-Parametric (Collapsed Gibbs Sampler)
% 1: GMM-EM Model Selection via BIC
% 2: CRP-GMM (Collapsed Gibbs Sampler)
est_options = [];
est_options.type             = 0;   % GMM Estimation Alorithm Type   

% If algo 0 or 2 selected:
est_options.samplerIter      = 50;  % Maximum Sampler Iterations                                 
est_options.do_plots         = 1;   % Plot Estimation Statistics
est_options.sub_sample       = 1;   % Size of sub-sampling of trajectories
                                    % 1/2 for 2D datasets, >2/3 for real    
% Metric Hyper-parameters
est_options.estimate_l       = 1;   % '0/1' Estimate the lengthscale, if set to 1
est_options.l_sensitivity    = 5;   % lengthscale sensitivity [1-10->>100]
est_options.length_scale     = [];  

% Fit GMM to Trajectory Data
% [Priors, Mu, Sigma] = fit_gmm(Xi_ref, Xi_dot_ref, est_options);
[Priors, Mu, Sigma] = fit_gmm(Data_mean(1:2,:), Data_mean(3:4,:), est_options);
K = length(Priors);

%% Generate GMM data structure for DS learning
clear ds_gmm; ds_gmm.Mu = Mu; ds_gmm.Sigma = Sigma; ds_gmm.Priors = Priors; 

%% (Recommended!) Step 2.1: Dilate the Covariance matrices that are too thin
% This is recommended to get smoother streamlines/global dynamics
adjusts_C  = 1;
if adjusts_C  == 1 
    if N == 2
        tot_dilation_factor = 1; rel_dilation_fact = 0.15;
    elseif N == 3
        tot_dilation_factor = 1; rel_dilation_fact = 0.75;        
    end
    Sigma_ = adjust_Covariances(ds_gmm.Priors, ds_gmm.Sigma, tot_dilation_factor, rel_dilation_fact);
    ds_gmm.Sigma = Sigma_;
end   

%% Align Covariance Matrices Re-estimate GMM parameters
unique_labels = unique(est_labels);
Mu_k = Mu;  Sigma_k = Sigma;
for k=1:length(unique_labels)
    cluster_points = Xi_ref(:,est_labels == unique_labels(k));
    if ~isempty(cluster_points)
        [ V_k, L_k, Mu_k(:,k) ] = my_pca( cluster_points );
        Sigma_k(:,:,k) = V_k*L_k*V_k';
    end
end
rel_dilation_fact = 0.5;
Sigma_k = adjust_Covariances(Priors, Sigma_k, 1, rel_dilation_fact);
ds_gmm.Mu = Mu_k; ds_gmm.Sigma = Sigma_k;

%%  Visualize Gaussian Components and labels on clustered trajectories 
% Extract Cluster Labels
[~, est_labels] =  my_gmm_cluster(Xi_ref, ds_gmm.Priors, ds_gmm.Mu, ds_gmm.Sigma, 'hard', []);

% Visualize Estimated Parameters
[h_gmm]  = visualizeEstimatedGMM(Xi_ref,  ds_gmm.Priors, ds_gmm.Mu, ds_gmm.Sigma, est_labels, est_options);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    Step 3: Estimate Local directions/attractors and Hyper-Plane Functions  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%  Create Radial-Basis Function around global attractor %%%%
B_r = 10;
radius_fun = @(x)(1 - my_exp_loc_act(B_r, att_g, x));
grad_radius_fun = @(x)grad_lambda_fun(x, B_r, att_g);

%%%%    Extract local linear DS directions and attractors   %%%%
% If attractors inside local regions decrease gauss_thres
gauss_thres = 0.25; plot_local_params = 1;
[att_l, local_basis] = estimate_local_attractors_lags(Data, est_labels, ds_gmm, gauss_thres, radius_fun, att_g);
if plot_local_params
    [h_att_l, h_dirs] = plotLocalParams(att_l, local_basis, Mu);
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Step 4: ESTIMATE CANDIDATE LYAPUNOV FUNCTION PARAMETERS  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Learn Single P_g for Initialization of WSAQF
[Vxf] = learn_wsaqf_lags(Data_sh);
P_g_prior = Vxf.P*K;

% Learn WSAQF 
scale_prior = 1; % ==> Check that it converges.. if it doesn't, reduce scale!
att_origin = att_l-repmat(att_g,[1 K]);
att_l_mod = att_l;

[Vxf]    = learn_wsaqf_lags(Data_sh, att_origin, P_g_prior*scale_prior);
P_g      = Vxf.P(:,:,1);
P_l      = Vxf.P(:,:,2:end);

%%% If any local attractor is equivalent to the global re-parametrize %%%
att_diff = att_l-repmat(att_g,[1 K]);
equal_g = find(any(att_diff)==0);
if equal_g ~= 0
    for ii=1:length(equal_g)
        P_g_ = P_g + P_l(:,:,equal_g(ii));
        P_l(:,:,equal_g(ii)) = P_g_;
    end
end
%% %%% Plot learned Lyapunov Function %%%%%
if N == 2
    contour = 1; % 0: surf, 1: contour
    clear lyap_fun_comb 
    
    % Lyapunov function
    lyap_fun_comb = @(x)lyapunov_function_combined(x, att_g, att_l_mod, 1, P_g, P_l, ds_gmm);
    title_string = {'$V(\xi) = (\xi-\xi_g^*)^TP_g(\xi-\xi_g^*) + \sum_{k=1}^K\beta^k((\xi-\xi_g^*)^TP_l^k(\xi-\xi_k^*))^2$'};
    if exist('h_lyap','var');  delete(h_lyap);     end
    if exist('hd_lyap','var');     delete(hd_lyap);     end
    [h_lyap] = plot_lyap_fct(lyap_fun_comb, contour, limits,  title_string, 0);
    [hd_lyap] = scatter(Data(1,:),Data(2,:),10,[1 1 0],'filled'); hold on;            
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%      Step 5: Learn Global DS Parameters     %%      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Choose Global DS type %%%
globalDS_type = 1; % 0: linear-DS
                   % 1: nonlinear-DS, LPV-DS learned with WSAQF Lyapunov Function 
%%% Choose Lyapunov Candidate type %%%                   
lyapType      = 1; % 0: P-QLF as in CoRL paper
                   % 1: WSAQF as in LAGS paper

clear global_ds
switch globalDS_type
    case 0 % Parametrize an agnostic global DS
    %%%%%%%%  Linear DS f_g(xi) = (Axi + b) %%%%%%%%
    
    globalDS_name = 'Linear Global DS';
    case 1 
    %%%%%%%%  LPV-DS f_g(xi) = sum_{k=1}^{K}\gamma_k(xi)(A_kxi + b_k) %%%%%%%%                
        switch lyapType
            case 0
                %%%%%%% Learn LPV-DS with P-QLF Constraint %%%%%
                % Type of constraints/optimization
                constr_type = 2; init_cvx    = 0;                                
                [A_g, b_g, ~] = optimize_lpv_ds_from_data(Data, att_g, constr_type, ds_gmm, P_g, init_cvx);                
                global_ds = @(x) lpv_ds(x, ds_gmm, A_g, b_g);                
                globalDS_name = 'LPV Global DS (P-QLF)';
                
            case 1
                %%%%%%% Learn LPV-DS with WSAQF Constraint %%%%%
                eps_scale = 100; %==> "If Constraints don't converge increase/decrease"
                enforce_g = 1;  %==> "Enforce A+A' on DS centered at global attractor"
                [A_g, b_g, ~] = optimize_globalDS_lags(Data, att_g, 3, ds_gmm, P_g, P_l, att_l_mod, eps_scale, enforce_g, equal_g);                                
                global_ds = @(x) lpv_ds(x, ds_gmm, A_g, b_g); 
                globalDS_name = 'LPV Global DS (WSAQF)';                                                              
        end        
end

%% %%%  Plot Resulting DS  %%%%%
% Fill in plotting options
ds_plot_options = [];
ds_plot_options.sim_traj  = 1;            % To simulate trajectories from x0_all
ds_plot_options.x0_all    = x0_all;       % Intial Points
ds_plot_options.init_type = 'ellipsoid';  % For 3D DS, to initialize streamlines
                                          % 'ellipsoid' or 'cube'  
ds_plot_options.nb_points = 30;           % No of streamlines to plot (3D)
ds_plot_options.plot_vol  = 1;            % Plot volume of initial points (3D)
% ds_plot_options.limits    = limits + [0 0.5 0 0];

[hd, hs, hr, x_sim] = visualizeEstimatedDS(Xi_ref, global_ds, ds_plot_options);
limits = axis;
title(globalDS_name, 'Interpreter','LaTex','FontSize',20)


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%      [OPTIONAL]: Evaluate Global DS Accuracy/Stability   %%      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%% Compare Velocities from Demonstration vs DS %%%%%
% Compute RMSE on training data
rmse = mean(rmse_error(global_ds, Xi_ref, Xi_dot_ref));
fprintf('Global-DS has prediction RMSE on training set: %d \n', rmse);

% Compute e_dot on training data
edot = mean(edot_error(global_ds, Xi_ref, Xi_dot_ref));
fprintf('Global-DS has e_dot on training set: %d \n', edot);

if exist('h_vel_comp','var'); delete(h_vel_comp); end
[h_vel_comp] = visualizeEstimatedVelocities(Data, global_ds);
title('Real vs. Estimated Velocities w/Global-DS', 'Interpreter', 'LaTex', 'FontSize', 15)

%% %%%  Plot Lyapunov function %%%%%
if N == 2
    clear lyap_der
    contour = 1; %0 :surface, 1:contour 
    switch lyapType
        case 0            
            % Lyapunov function
            lyap_fun = @(x)lyapunov_function_PQLF(x, att_g, P_g);
            % Plots
            title_string = {'$V(\xi) = (\xi-\xi^*)^TP(\xi-\xi^*)$'};
            h_lyap     = plot_lyap_fct(lyap_fun, contour, limits,  title_string, 0);
            [hddd] = scatter(Data(1,:),Data(2,:),10,[1 1 0],'filled'); hold on;
    
            % Derivative of Lyapunov function (gradV*f(x))
            lyap_der = @(x)lyapunov_derivative_PQLF(x, att_g, P_g, global_ds);
            title_string_der = {'Lyapunov Function Derivative $\dot{V}(\xi)$'};
        case 1
            % Derivative of Lyapunov function (gradV*f(x))
            lyap_der = @(x)lyapunov_combined_derivative(x, att_g, att_l_mod, global_ds, 1, P_g, P_l);                       
            
            title_string_der = {'Lyapunov Function Derivative $\dot{V}_{DD-WSAQF}(\xi)$'};
    end
    if exist('h_lyap_der','var'); delete(h_lyap_der); end
    if exist('hd_lyap_der','var'); delete(hd_lyap_der); end
    [h_lyap_der] = plot_lyap_fct(lyap_der, contour, limits,  title_string_der, 1);
    [hd_lyap_der] = scatter(Data(1,:),Data(2,:),10,[1 1 0],'filled'); hold on;            
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%      [OPTIONAL]: Simulate Global DS on Robot with Passive-DS Ctrl   %%      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set up a simple robot and a figure that plots it
robot = create_simple_robot();
fig1 = initialize_robot_figure(robot);
title('Feasible Robot Workspace','Interpreter','LaTex')
% Base Offset
base = [-1 1]';
% Axis limits
limits = [-2.5 0.75 -0.45 2.2];
axis(limits)
if exist('hds_rob','var');  delete(hds_rob);  end
if exist('hdata_rob','var');delete(hdata_rob);end
if exist('hatt_rob','var');delete(hatt_rob);end
[hds_rob] = plot_ds_model(fig1, global_ds, [0;0], limits,'medium'); hold on;
[hdata_rob] = scatter(Data(1,:),Data(2,:),10,[1 0 0],'filled'); hold on;            
[hatt_rob] = scatter(att_g(1),att_g(2), 150, [0 0 0],'d','Linewidth',2); hold on;
title(globalDS_name, 'Interpreter','LaTex','FontSize',20)
    xlabel('$\xi_1$','Interpreter','LaTex','FontSize',20);
    ylabel('$\xi_2$','Interpreter','LaTex','FontSize',20);
    
%% Simulate Passive DS Controller function
dt = 0.01;
show_DK = 0;
if show_DK
    struct_stiff = [];
    struct_stiff.DS_type  = 'global'; % Options: 'global', 'lags'
    struct_stiff.gmm      = ds_gmm;
    struct_stiff.A_g      = A_g;
    struct_stiff.basis    = 'D'; % Options: D (using the same basis as D) or I (using the world basis)
    struct_stiff.L        = [0.5 0; 0 0.5]; % Eigenvalues for Daming matrix
    struct_stiff.mod_step = 50; % Visualize D/K matrices every 'mod_step'     
    
    % Select this one if we want to see the Damping and Stiffness matrices
    simulate_passiveDS(fig1, robot, base, global_ds, att_g, dt, struct_stiff);
    
else
    simulate_passiveDS(fig1, robot, base, global_ds, att_g, dt);
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Step 6: Create Activation Function (i.e. select locallly active regions)     %%      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Visualize Global DS with Locally Linear Partitions
clc;
ds_plot_options.sim_traj  = 0;            
[hd, hs, hr, x_sim, fig_ds] = visualizeEstimatedDS(Data_mean(1:2,:), global_ds, ds_plot_options);
title('Choose regions to locally activate:', 'Interpreter','LaTex','FontSize',20);
[ h_gmm, h_ctr, h_txt ] = plotLLRegions(ds_gmm.Mu, ds_gmm.Sigma);

% Compute Hyper-plane functions per local model
w = zeros(2,K); breadth_mod = 50;
h_functor = cell(1,K); grad_h_functor = cell(1,K);
lambda_functor  = cell(1,K); Data_k = [];
for k=1:K
    % Create Hyper-Plane functions
%     w(:,k) =  (Mu(:,k)-att_l(:,k))/norm(Mu(:,k)-att_l(:,k));
    w(:,k) = local_basis(:,1,k);
    h_functor{k} = @(x)hyper_plane(x,w(:,k),att_l(:,k));
    grad_h_functor{k} = @(x)grad_hyper_plane(x,w(:,k),h_functor{k});
    lambda_functor{k} = @(x)(1-my_exp_loc_act(breadth_mod, att_l(:,k), x));
    grad_lambda_functor{k} = @(x)grad_lambda_fun(x, breadth_mod, att_l(:,k));    
end

%% Choose Locally Linear Regions that you wanto to activate
choose_k = 1;
choosen_active = 1:K;
choosen_active = [4 8 5];
% choosen_active = [1 2 3 6 9 7];
Data_ = Data_mean(1:2,:);
[~, est_labels_] =  my_gmm_cluster(Data_mean(1:2,:), ds_gmm.Priors, ds_gmm.Mu, ds_gmm.Sigma, 'hard', []);

if choose_k
    Data_ = [];
    for k_=1:length(choosen_active)
        Data_ = [Data_ Data_mean(1:2,est_labels_==choosen_active(k_))];
    end
end

%% === Construct Activation Function ===
act_type = 1;

% Radial Basis Function centered at the global attractor
B_r = 10;
radius_fun = @(x)(1 - my_exp_loc_act(B_r, att_g, x));
grad_radius_fun = @(x)grad_lambda_fun(x, B_r, att_g);

% Train GMR/GPR model of the data \Xi_mean,\Kappa
X_train = Data_; y_train = ones(1,length(X_train));
switch act_type
    case 0 % With GMR
        % Option 1: with normalized pdf of gmm
        
        
        % GMM-PDF
        gmm_fun    = @(x)ml_gmm_pdf(x, ds_gmm.Priors, ds_gmm.Mu, ds_gmm.Sigma);
        activ_fun  = @(x) 1 - gmm_fun(x);
        
        % Direct vs Max. Alpha functions
        alpha_fun  = @(x)( (1-radius_fun(x))'.*max(0,activ_fun(x))' + radius_fun(x)');
        
        % Compute gradient       
        grad_gmm_fun   = @(x)grad_gmm_pdf(x, ds_gmm);
        grad_alpha_fun = @(x)gradient_alpha_fun(x,radius_fun, grad_radius_fun, gmm_fun, grad_gmm_fun, 'gmm');
        activation_name = 'GMR';
        
    case 1 % With GPR       
        epsilon = 0.1; rbf_var = 0.2; % assuming output variance = 1
        model.X_train   = X_train';   model.y_train   = y_train';
        gpr_fun    = @(x) my_gpr(x',[],model,epsilon,rbf_var);
        activ_fun  = @(x) 1 - gpr_fun(x);                    
        alpha_fun  = @(x)( (1-radius_fun(x))'.*activ_fun(x)' + radius_fun(x)');        
        
        % Compute gradient       
        grad_gpr_fun   = @(x)gradient_gpr(x, model, epsilon, rbf_var);
        grad_alpha_fun = @(x)gradient_alpha_fun(x,radius_fun, grad_radius_fun, gpr_fun, grad_gpr_fun, 'gpr');        
        activation_name = 'GPR';        
end

%% Plot values of mixing function to see where transition occur
figure('Color',[1 1 1])
[h_act] = plot_mixing_fct_2d(limits, alpha_fun); hold on;
[hds_rob] = plot_ds_model(h_act, global_ds, [0;0], limits,'medium'); hold on;
[hdata_act] = scatter(Data(1,:),Data(2,:),10,[0 0 0],'filled'); hold on;            
[hatt_act] = scatter(att_g(1),att_g(2), 150, [0 0 0],'d','Linewidth',2); hold on;
[ h_gmm, h_ctr, h_txt ] = plotLLRegions(ds_gmm.Mu, ds_gmm.Sigma, choosen_active);
[h_att_l, h_dirs] = plotLocalParams(att_l, local_basis, Mu, choosen_active);
xlabel('$\xi_1$','Interpreter', 'LaTex','FontSize',15)
ylabel('$\xi_2$','Interpreter', 'LaTex','FontSize',15)
title_name = strcat(activation_name, '-based Activation function');
title(title_name, 'Interpreter','LaTex','FontSize',20);

%% Visualize Activation Function in 3D
% Visualize a function
eval_fun       = @(x)alpha_fun(x);
h_gamma      = plot_lyap_fct(eval_fun, 0, limits,  {'$\alpha(\xi)$ Function'}, 0);
[hdata_act] = scatter(Data(1,:),Data(2,:),10,[1 0 0],'filled'); hold on;            
% Visualize its gradient
% grad_eval_fun  = @(x)grad_alpha_fun(x);
% h_grad_gamma = plot_gradient_fct(grad_eval_fun, limits,  '$r$ and $\nabla_{\xi}r$ Function');

