%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demo Script for GMM-based LAGS-DS Learning introduced in paper:         %
%  'Locally Active Globally Stable Dynamical Systems';                    %
% N. Figueroa and A. Billard; TRO/IJRR 2019                               %
% With this script you can draw simple linear trajectories and learn      %
% a corresponding linear LAGS-DS
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
est_options.do_plots         = 0;   % Plot Estimation Statistics
est_options.sub_sample       = 2;   % Size of sub-sampling of trajectories
% Metric Hyper-parameters
est_options.estimate_l       = 1;   % '0/1' Estimate the lengthscale, if set to 1
est_options.l_sensitivity    = 0.5;   % lengthscale sensitivity [1-10->>100]
est_options.length_scale     = [];  

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
grad_h_fun = @(x)grad_hyper_plane(x,w,h_fun);

% Gaussian Covariance scaling


%%%%%%%%%% Construct Activation function %%%%%%%%%%
gauss_opt_thres = 0.25; 
c_rad       = 50;
radius_fun  = @(x)(1 - my_exp_loc_act(c_rad, att_g, x));

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

% Compute gradients
grad_radius_fun = @(x)grad_lambda_fun(x, c_rad, att_g);
hess_radius_fun = @(x) hess_lambda_fun(x, c_rad, att_g);
gauss_fun       = @(x)my_gaussPDF(x, Mu, Sigma);
grad_gauss_fun  = @(x)grad_gauss_pdf(x, Mu, Sigma);
grad_alpha_fun  = @(x)((1/Norm).*gradient_alpha_fun(x,radius_fun, grad_radius_fun, gauss_fun, grad_gauss_fun, 'gauss'));

%% Plot values of activation function to see where transition occur
close all;
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
[h_data, h_att, h_vel] = plot_reference_trajectories_on_DS(Data, att_g, vel_samples, vel_size, fig1);
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

% Learn Single P_g for Initialization of WSAQF
[Vxf] = learn_wsaqf_lags(Data);
P_g_prior = Vxf.P;

% Learn WSAQF 
scale_prior = 1; % ==> Check that it converges.. if it doesn't, reduce scal1!
att_shifted = att_l-att_g;
[Vxf]    = learn_wsaqf_lags(Data, att_shifted, P_g_prior*scale_prior);
P_g      = Vxf.P(:,:,1);
P_l      = Vxf.P(:,:,2:end);

trace(P_g)
trace(P_l(:,:,1))

%%% If any local attractor is equivalent to the global re-parametrize %%%
att_diff = att_l- att_g;
equal_g = find(any(att_diff)==0);
if equal_g ~= 0
    for ii=1:length(equal_g)
        P_g_ = P_g + P_l(:,:,equal_g(ii));
        P_l(:,:,equal_g(ii)) = P_g_;
    end
end

contour = 0; % 0: surf, 1: contour
clear lyap_fun_comb

% Lyapunov function
lyap_fun_comb = @(x)lyapunov_function_combined(x, att_g, att_l, 1, P_g, P_l, ds_gmm);
title_string = {'$V(\xi) = (\xi-\xi_g^*)^TP_g(\xi-\xi_g^*) + \sum_{k=1}^K\beta^k((\xi-\xi_g^*)^TP_l^k(\xi-\xi_k^*))^2$'};
if exist('h_lyap','var');  delete(h_lyap);     end
if exist('hd_lyap','var');     delete(hd_lyap);     end
[h_lyap] = plot_lyap_fct(lyap_fun_comb, contour, limits,  title_string, 0);
[hd_lyap] = scatter(Data(1,:),Data(2,:),10,[1 1 0],'filled'); hold on;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%% Step 3: ESTIMATE Global and Local System Matrices     %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%  GLOBAL DS PARAMETER INITIALIZATION %%%%%%%%%%%%%%
% Global Dynamics type
fg_type = 1;          % 0: Fixed Linear system Axi + b
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

% Stability Checks -- lambda_min < -1 for good estimation
Q_g  = A_g'*P_g + P_g*A_g;
lambda_Qg = eig(Q_g)
Q_gl = A_g'*P_l';
lambda_Qgl = eig(Q_gl)

% Check \dot(Vg)
lyap_der_glob = @(x)lyapunov_combined_derivative_global(x, att_g, att_l, P_g, P_l, A_g);
if exist('h_lyap_der_glob','var'); delete(h_lyap_der_glob); end
h_lyap_der_glob = plot_lyap_fct(lyap_der_glob, 0, limits_,  'Lyapunov Derivative of Global DS', 1);

% Local Attractor diffusive function component
breadth_mod = 20; % larger number.. smaller the rbf width
lambda_fun = @(x)lambda_mod_fun(x, breadth_mod, att_l, grad_h_fun, grad_lyap_fun);
grad_lambda_fun = @(x)grad_lambda_fun(x, breadth_mod, att_l);
%%% TODO -> compute epsilon from maximum of this function in the compact set
%%% within Brl

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Option 1: Unconstrained Local-Dynamics Estimation, given known \kappa %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
kappa = 1
[A_l, b_l, A_d, b_d] = estimate_localDS_known_gamma(Data, A_g,  att_g, att_l, fl_type, kappa,w, P_g, P_l, Q);
Lambda_l = eig(A_l)
kappa = max(abs(Lambda_l))/min(abs(Lambda_l))

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Option 2: (Un)-Constrained Local-Dynamics Estimation, estimating \kappa %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%  LOCAL DS OPTIMIZATION OPTIONS %%%%%%%%%%%%%%%%%%%%%
% Construct variables and function handles for stability constraints
clear stability_vars 
stability_vars.solver          = 'fmincon'; % options: 'baron' or 'fmincon'
stability_vars.grad_h_fun      = grad_h_fun;
stability_vars.add_constr      = 1; % 0: no stability constraints
                                    % 1: Adding sampled stability constraints                                   
% Type of constraint to evaluate
stability_vars.constraint_type = 'matrix';  % options: 'full/matrix/hessian'
stability_vars.epsilon         = 1e-4;      % small number for f_Q < -eps
stability_vars.do_plots        = 1;         % plot current lyapunov constr. 
stability_vars.init_samples    = 100;        % Initial num/boundary samples
stability_vars.iter_samples    = 50;        % Initial num/boundary samples

% Function handles for contraint evaluation
stability_vars.alpha_fun     = alpha_fun;
stability_vars.activ_fun     = activ_fun;
stability_vars.h_fun         = h_fun;
stability_vars.lambda_fun    = lambda_fun;    

% Variable for different type constraint types
if strcmp(stability_vars.constraint_type,'full')
    stability_vars.grad_lyap_fun = grad_lyap_fun;
else
    stability_vars.P_l           = P_l;
    stability_vars.P_g           = P_g;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if stability_vars.add_constr
    % Draw Initial set of samples for point-wise stability constraints
    desired_samples       = stability_vars.init_samples;
    desired_alpha_contour = 0.99;
    desired_Gauss_contour = -Norm*(desired_alpha_contour-1);
    chi_samples = draw_chi_samples (Sigma,Mu,desired_samples, stability_vars.activ_fun, 'isocontours', desired_Gauss_contour);
    
    % Local DS Optimization with Constraint Rejection Sampling
    stability_ensured = 0; iter = 1; test_grid = 0;
    while ~stability_ensured
        if iter == 1
            fprintf('First iteration, using %d BOUNDARY chi-samples..\n', size(chi_samples,2));
        else
            fprintf('iter = %d, using %d chi-samples..\n', iter, size(chi_samples,2));
        end
        pause(1);
        
        % Feed chi_samples to optimization structure
        stability_vars.chi_samples = chi_samples;
        % Run optimization with chi-sample constraints
        if strcmp(stability_vars.constraint_type,'hessian')
            % Approximate - very conservative estimation
            [A_l, b_l, A_d, b_d, gamma] = optimize_localDS_for_LAGS_Hess(Data, A_g, att_g, stability_vars);
        else
            % Full condition - less conservative estimation
            [A_l, b_l, A_d, b_d] = optimize_localDS_for_LAGS(Data, A_g, att_g, fl_type, stability_vars);
        end
        
        % Create function handles for current estimate of f_Q, grad_fQ
        clear f_Q
        f_Q = @(x)fQ_constraint_single(x, att_g, att_l, P_g, P_l, alpha_fun, h_fun, grad_h_fun, A_g, A_l, A_d);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%  Check for violations in compact set by grid-sampling  %%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Plot Current Lyapunov Constraints function fQ
        if stability_vars.do_plots
            contour = 0;
            if exist('h_lyap','var'); delete(h_lyap);  end
            h_lyap = plot_lyap_fct(f_Q, contour, limits,'Test $f_Q(\xi)$ with Grid-Sampling',1);        hold on;
            if exist('h_samples_used','var'); delete(h_samples_used);  end
            if contour
                h_samples_used = scatter(chi_samples(1,:),chi_samples(2,:),'+','c');
            else
                h_samples_used = scatter3(chi_samples(1,:),chi_samples(2,:),f_Q(chi_samples),'+','c');
            end
        end
        cv_options                   = [];
        cv_options.chi_params        = struct('Mu',Mu,'Sigma',Sigma);
        cv_options.num_samples       = 10^5;
        cv_options.epsilon           = stability_vars.epsilon;
        tic;
        [constraint_violations_grid, max_fQ_grid, max_X_grid] = search_constraintViolations(f_Q, activ_fun, cv_options);
        toc;
        
        % If we still have violations sample new contraints-eval points
        new_samples = []; num_grid_violations = length(constraint_violations_grid);
        if max_fQ_grid > -stability_vars.epsilon
            fprintf(2, 'Maxima in compact set is positive (f_max=%2.8f)! Current form is Not Stable!\n', max_fQ_grid);
            if num_grid_violations <= stability_vars.iter_samples
                new_samples = constraint_violations_grid;
            else
                num_max_samples  = round(stability_vars.iter_samples/2);
                num_rand_samples = stability_vars.iter_samples - num_max_samples;               
                new_samples = [new_samples constraint_violations_grid(:,1:num_max_samples)];
                constraint_violations_grid(:,1:num_max_samples) = [];
                new_samples = [new_samples constraint_violations_grid(:,randsample(length(constraint_violations_grid),num_rand_samples))];
            end
            if stability_vars.do_plots
                if exist('h_new_samples','var'); delete(h_new_samples);  end
                if contour
                    h_new_samples = scatter(new_samples(1,:),new_samples(2,:),'+','r');
                else
                    h_new_samples = scatter3(new_samples(1,:),new_samples(2,:),f_Q(new_samples),'+','r');
                end
            end
            chi_samples = [chi_samples new_samples];
        else
            
            fprintf('Maxima in compact set is negative (f_max=%2.8f)! Stability is ensured!\n', max_fQ_grid);
            test_grid = 1;
            
            % Check via optimization if parameters are stable
            if test_grid
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%% DOUBLE CHECK! Find maxima in compact set with Gradient Ascent on estimated fQ() %%%%
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Gradient of Lyapunov Function
                clear grad_fQ
                grad_fQ = @(x)gradient_fQ_constraint_single(x, att_g, att_l, P_g, P_l, alpha_fun, grad_alpha_fun, h_fun, grad_h_fun, A_g, A_l, A_d);
                % Plot Current Lyapunov Constraints function fQ
                if stability_vars.do_plots
                    plot_lyap_fct(f_Q, 1, limits, 'Estimated $f_Q(\xi)$',1); hold on;
                    plot_gradient_fct(grad_fQ, limits,  'Estimated $f_Q$ with $\nabla_{\xi}f_Q$');
                end
                % Do the maxima search with gradient ascent
                lm_options                = [];
                lm_options.type           = 'grad_ascent';
                lm_options.num_ga_trials  = 10;
                lm_options.do_plots       = stability_vars.do_plots;
                lm_options.init_set       = chi_samples;
                lm_options.verbosity      = 0;
                [local_max, local_fmax]   = find_localMaxima(f_Q, grad_fQ, lm_options);
                [max_val, max_id]         = max(local_fmax);
                
                % Compare f_Q - mod_term < 0                
                if max_val > 0.05
                    fprintf (2, 'There was an fQ_max(%2.8f) > 0 at x=%3.3f,y=%3.3f found :( !!\n', max_val, local_max(1,max_id),local_max(2,max_id));
                    fprintf (2, 'You must re-run optimization, try increasing epsilon! \n');
                else
                    fprintf ('+++++ ALL fQ_max < 0 +++++!!\n');
                    % Estimated parameters ensure stability as fQ_max < 0
                    stability_ensured = 1;
                end
                fprintf('Optimization converged to a stable solution with %d chi_samples in %d iterations!\n', size(chi_samples,2),iter);
            end
        end
        
        % Constraint Sampling Loop
        iter = iter + 1;
    end
    
else
    tic;
    [A_l, b_l, A_d, b_d] = optimize_localDS_for_LAGS(Data, A_g, att_g, fl_type, stability_vars);
    toc;
end

Lambda_l = eig(A_l)
kappa = max(abs(Lambda_l))/min(abs(Lambda_l))

%% Constraint Checks
Q_l  = A_l'*P_l
eig(Q_l+Q_l')
Q_lg = 2*A_l'*P_g
eig(Q_lg+Q_lg')

% Function for DS
lagsDS_linear = @(x) lags_ds(att_g, x, mix_type, alpha_fun, A_g, b_g, A_l, b_l, att_l, h_fun, A_d, b_d, lambda_fun, grad_h_fun);
% x0_all = [data{1}(1:2,1) data{2}(1:2,1)];
x0_all = [data{1}(1:2,1) [-1.656;0.5822] [-1.8;0.655] [-1.3;1.43] ];

%% %%%  Plot Resulting DS  %%%%%
% Fill in plotting options
ds_plot_options = [];
ds_plot_options.sim_traj = 1;            % To simulate trajectories from x0_all
ds_plot_options.x0_all   = x0_all;       % Intial Points
ds_plot_options.limits   = limits;       % Plot volume of initial points (3D)

% Plot Mixing function
% h_dec = plot_mixing_fct_2d(limits, alpha_fun); hold on;
[hd, hs, hr, x_sim] = visualizeEstimatedDS(Xi_ref, lagsDS_linear, ds_plot_options);
title('Linear LAGS-DS with Single-Active Region', 'Interpreter','LaTex','FontSize',20)

text(att_g(1),att_g(2),'$\mathbf{\xi}^*_g$','Interpreter', 'LaTex','FontSize',15); hold on;
h_att = scatter(att_l(1),att_l(2),150,[0 0 0],'d','Linewidth',2); hold on;


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%      [OPTIONAL]: Evaluate Global DS Accuracy/Stability   %%      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%% Compare Velocities from Demonstration vs DS %%%%%
% Compute RMSE on training data
rmse = mean(rmse_error(lagsDS_linear, Xi_ref, Xi_dot_ref));
fprintf('Global-DS has prediction RMSE on training set: %d \n', rmse);

% Compute e_dot on training data
edot = mean(edot_error(lagsDS_linear, Xi_ref, Xi_dot_ref));
fprintf('Global-DS has e_dot on training set: %d \n', edot);

if exist('h_vel_comp','var'); delete(h_vel_comp); end
[h_vel_comp] = visualizeEstimatedVelocities(Data, lagsDS_linear);
title('Real vs. Estimated Velocities w/Global-DS', 'Interpreter', 'LaTex', 'FontSize', 15)

%% %%%  Plot Lyapunov function Derivative %%%%%
if N == 2
    clear lyap_der
    contour = 0; %0 :surface, 1:contour
    % Derivative of Lyapunov function (gradV*f(x))
    %         lyap_der = @(x)lyapunov_combined_derivative(x, att_g, att_l, ds_lags_single, lyap_type, P_g, P_l, beta_eps);
    
    % Derivative of Lyapunov function (fully factorized)
    clear  lyap_der
    lyap_der = @(x)lyapunov_combined_derivative_full(x, att_g, att_l, P_g, P_l, alpha_fun, h_fun, lambda_fun, grad_h_fun, A_g, A_l, A_d);
    title_string = {'$V(\xi) = (\xi-\xi_g^*)^TP_g(\xi-\xi_g^*) + \beta((\xi-\xi_g^*)^TP_l(\xi-\xi_l^*))^2$'};
    title_string_der = {'Lyapunov Function Derivative $\dot{V}(\xi)$'};
    
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
if exist('h_act','var');delete(h_act);end
[h_act] = plot_mixing_fct_2d(limits, alpha_fun); hold on;
[hds_rob] = plot_ds_model(fig1, lagsDS_linear, [0;0], limits,'medium'); hold on;
[hdata_rob] = scatter(Data(1,:),Data(2,:),10,[1 0 0],'filled'); hold on;            
[hatt_rob] = scatter(att_g(1),att_g(2), 150, [0 0 0],'d','Linewidth',2); hold on;
title('Linear LAGS-DS', 'Interpreter','LaTex','FontSize',20)
    xlabel('$\xi_1$','Interpreter','LaTex','FontSize',20);
    ylabel('$\xi_2$','Interpreter','LaTex','FontSize',20);
    
%% Simulate Passive DS Controller function
dt = 0.01;
show_DK = 1;
if show_DK
    struct_stiff = [];
    struct_stiff.DS_type             = 'global'; % Options: 'global', 'lags'
    struct_stiff.gmm                 = ds_gmm;
    struct_stiff.A_g                 = A_g;
    % Modify options to show linear lags-ds K correctly
    struct_stiff.A_l                 = A_l;
    struct_stiff.A_d                 = A_d;
    struct_stiff.att_k               = att_l;
    struct_stiff.alpha_fun           = alpha_fun;
    struct_stiff.grad_alpha_fun      = grad_alpha_fun;
    struct_stiff.h_functor           = h_fun;
    struct_stiff.grad_h_functor      = grad_h_fun;
    struct_stiff.lambda_functor      = lambda_fun;
    struct_stiff.grad_lambda_functor = grad_lambda_fun;
    struct_stiff.basis               = 'D';  % Options: D (using the same basis as D) or I (using the world basis)
    struct_stiff.L                   = [1 0; 0 1]; % Eigenvalues for Daming matrix
    struct_stiff.mod_step            = 2; % Visualize D/K matrices every 'mod_step'     
    
    % Select this one if we want to see the Damping and Stiffness matrices
    simulate_passiveDS(fig1, robot, base, lagsDS_linear, att_g, dt, struct_stiff);
    
else
    simulate_passiveDS(fig1, robot, base, lagsDS_linear, att_g, dt);
end
