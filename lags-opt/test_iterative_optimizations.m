%% Testing un-constrained iterative optimization implementations
%% Define Functions, Gradient and Hessian.. using Lyapunov function found
% from other script (uses variable defined in demo_lagsDS_single.m)
f      = @(x)(-lyapunov_function_combined(x, att_g, att_l, 1, P_g, P_l)); 
grad_f = @(x)(-gradient_lyapunov(x, att_g, att_l, P_g, P_l));
hess_f = @(x)(-hessian_lyapunov(x, att_g, att_l, P_g, P_l));

%% Define Functions, Gradient and Hessian.. using f_Q
f      = @(x)fQ_constraint_single(x, att_g, att_l, P_g, P_l, alpha_fun, h_fun, grad_h_fun, A_g, A_l, A_d);
grad_f = @(x)gradient_fQ_constraint_single(x, att_g, att_l, P_g, P_l, alpha_fun, grad_alpha_fun, h_fun, grad_h_fun, A_g, A_l, A_d);

% Plot Current Lyapunov Constraints function fQ
% chi_min = min(chi_samples,[],2);
% chi_max = max(chi_samples,[],2);
% limits = [chi_min(1) chi_max(1) chi_min(2) chi_max(2)];
plot_lyap_fct(f,1,limits_,'$f_Q(\xi)$',1);        hold on;

% Functions to evaluate
eval_fun       = @(x)alpha_fun(x);
plot_gradient_fct(grad_f, limits_,  '$f_Q$ and $\nabla_{\xi}f_Q$ Function');

%% OPTION 1: Maxima Search using Gradient Ascent
ga_options = [];
ga_options.gamma    = 0.001;  % step size (learning rate)
ga_options.max_iter = 1500;   % maximum number of iterations
ga_options.f_tol    = 1e-10;  % termination tolerance for F(x)
ga_options.plot     = 1;      % plot init/final and iterations
ga_options.verbose  = 0;      % Show values on iterations

% Set Initial value
x0 = Mu;
x0 = chi_samples_used(:,50+randsample(2*desired_samples,1));
fprintf('Finding maxima in Test function using Gradient Ascent...\n');

% Plot the Lyapunov function to find minima
% plot_lyap_fct(f, 1, limits_,'Test $f(\xi)$',1);

[f_max, x_max, fvals, xvals, h_points] = gradientAscent(f,grad_f, x0, ga_options);

%% OPTION 2: Maxima Search using Newton Method
nm_options = [];
nm_options.max_iter = 10;   % maximum number of iterations
nm_options.f_tol    = 1e-10;  % termination tolerance for F(x)
nm_options.plot     = 1;      % plot init/final and iterations
nm_options.verbose  = 1;      % Show values on iterations

% Set Initial value
% x0 = Mu;
x0 = chi_samples_used(:,randsample(length(chi_samples_used),1));
fprintf('Finding maxima in Chi using Newton Method...\n');

% Plot the Lyapunov function to find minima
plot_lyap_fct(f, 1, limits_,'Test $f(\xi)$',1);

% Create hessian handle of surrogate function
[f_max, x_max, fvals, xvals, h_points] = newtonMethod(f, grad_f, hess_f, x0, nm_options);