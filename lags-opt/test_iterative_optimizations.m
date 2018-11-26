%% Testing un-constrained iterative optimization implementations
%% Function Option 1: Define Functions, Gradient and Hessian.. using Lyapunov function found
% from other script (uses variable defined in demo_lagsDS_single.m)
clear f grad_f hess_f
f      = @(x)(-lyapunov_function_combined(x, att_g, att_l, 1, P_g, P_l)); 
grad_f = @(x)(-gradient_lyapunov(x, att_g, att_l, P_g, P_l));
hess_f = @(x)(-hessian_lyapunov(x, att_g, att_l, P_g, P_l));

% Plot the function
plot_lyap_fct(f,1,limits_,'$f_Q(\xi)$',0);        hold on;

% Plot the gradient of the function
plot_gradient_fct(grad_f, limits_,  '$f_Q$ and $\nabla_{\xi}f_Q$ Function');

%% Function Option 2: Define Functions, Gradient (and Hessian?).. using Single Gaussian Distribution
clear f grad_f hess_f
f       = @(x)(my_gaussPDF(x, Mu, Sigma));
grad_f  = @(x)(grad_gauss_pdf(x, Mu, Sigma));
hess_f  = @(x)(hess_gauss_pdf(x, Mu, Sigma));

% Plot the function
plot_lyap_fct(f,1,limits_,'$f(\xi)$',1);        hold on;

% Plot the gradient of the function
plot_gradient_fct(grad_f, limits_,  '$N$ and $\nabla_{\xi}N$ Function');

%% Function Option 3: Define Functions, Gradient (and Hessian?).. using alpha_fun (Gaussian-based)
clear f grad_f hess_f
f       = @(x)( (1-radius_fun(x)).*activ_fun(x)' + radius_fun(x));
grad_f  = @(x)((1/Norm).*gradient_alpha_fun(x,radius_fun, grad_radius_fun, gauss_fun, grad_gauss_fun, 'gauss'));

% Plot the function
plot_lyap_fct(f,1,limits_,'$f(\xi)$',0);        hold on;

% Plot the gradient of the function
plot_gradient_fct(grad_f, limits_,  '$\alpha$ and $\nabla_{\xi}\alpha$ Function');

%% Function Option 4: Define Functions, Gradient (and Hessian?).. using f_Q
clear f grad_f hess_f
f      = @(x)(fQ_constraint_single(x, att_g, att_l, P_g, P_l, alpha_fun, h_fun, grad_h_fun, A_g, A_l, A_d));
grad_f = @(x)(gradient_fQ_constraint_single(x, att_g, att_l, P_g, P_l, alpha_fun, grad_alpha_fun, h_fun, grad_h_fun, A_g, A_l, A_d));

% Plot the function
plot_lyap_fct(f,1,limits_,'$f_Q(\xi)$',1);        hold on;

% Plot the gradient of the function
plot_gradient_fct(grad_f, limits_,  '$f_Q$ and $\nabla_{\xi}f_Q$ Function');

%% --- TODO --- Function Option 5: Define Functions, Gradient (and Hessian?).. using GMM  
% ...........
% ...........
% ...........

%% Test the maxima search function
lm_options                   = [];
lm_options.type              = 'grad_ascent';
lm_options.num_ga_trials     = 10;
lm_options.do_plots          = 1;
lm_options.init_set          = chi_samples;
lm_options.verbosity         = 0;
[local_max, local_fmax]      =  find_localMaxima(f, grad_f, lm_options);
if any(local_fmax >= 0)
    fprintf (2, 'ALL fQ_max < 0 !!\n');
else
    fprintf ('+++++ ALL fQ_max < 0 +++++!!\n');
end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%      Optimization OPTION 1: Maxima Search using Gradient Ascent       %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ga_options = [];
ga_options.gamma    = 0.001;  % step size (learning rate)
ga_options.max_iter = 1000;    % maximum number of iterations
ga_options.f_tol    = 1e-10;   % termination tolerance for F(x)
ga_options.plot     = 1;       % plot init/final and iterations
ga_options.verbose  = 1;       % Show values on iterations

% Set Initial value
x0 = chi_samples(:,randsample(length(chi_samples),1));
fprintf('Finding maxima in Test function using Gradient Ascent...\n');
[f_max, x_max, fvals, xvals, h_points] = gradientAscent(f,grad_f, x0, ga_options);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%      Optimization OPTION 2: Maxima Search using Newton Method         %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% CANNOT HANDLE INDEFINITE HESSIANS! Only quadratic functions or convex
nm_options = [];
nm_options.max_iter = 100;   % maximum number of iterations
nm_options.f_tol    = 1e-10;  % termination tolerance for F(x)
nm_options.plot     = 1;      % plot init/final and iterations
nm_options.verbose  = 1;      % Show values on iterations

% Set Initial value
x0 = chi_samples(:,randsample(length(chi_samples),1));
fprintf('Finding maxima in Chi using Newton Method...\n');

% Plot the Lyapunov function to find minima
% plot_lyap_fct(f, 1, limits_,'Test $f(\xi)$',1);

% Create hessian handle of surrogate functionc
clc;
[f_max, x_max, fvals, xvals, h_points] = newtonMethod(f, grad_f, hess_f, x0, nm_options);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Optimization OPTION 3: Maxima Search using Conjugate Gradient Method  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% CANNOT HANDLE INDEFINITE HESSIANS! Only quadratic functions or convex
cga_options = [];
cga_options.max_iter = 50;   % maximum number of iterations
cga_options.f_tol    = 1e-10;  % termination tolerance for F(x)
cga_options.plot     = 1;      % plot init/final and iterations
cga_options.verbose  = 1;      % Show values on iterations

% Set Initial value
x0 = chi_samples(:,randsample(length(chi_samples),1));
fprintf('Finding maxima in Chi using Newton Method...\n');

% Plot the Lyapunov function to find minima
% plot_lyap_fct(f, 1, limits_,'Test $f(\xi)$',1);

% Create hessian handle of surrogate functionc
clc;
[f_max, x_max, fvals, xvals, h_points] = conjugateGradientMethod(f, grad_f, hess_f, x0, cga_options);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Optimization OPTION 4: Maxima Search using Nonlinear Conjugate Gradient (line-search) Method  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% CURRENTLY ONLY WORKS FOR THE FIRST FUNCTION TYPE!!!
ls_options = [];
ls_options.max_iter = 500;   % maximum number of iterations
ls_options.f_tol    = 1e-10;  % termination tolerance for F(x)
ls_options.plot     = 1;      % plot init/final and iterations
ls_options.verbose  = 1;      % Show values on iterations

% Set Initial value
x0 = chi_samples(:,randsample(length(chi_samples),1));
fprintf('Finding maxima in Chi using Newton Method...\n');

% Plot the Lyapunov function to find minima
% plot_lyap_fct(f, 1, limits_,'Test $f(\xi)$',1);

% Create hessian handle of surrogate functionc
clc;
[f_max, x_max, fvals, xvals, h_points] = lineSearchMethod(f, grad_f, x0, ls_options);


%% Try finding the maxima with fminunc

% Set Initial value
x0 = chi_samples(:,randsample(length(chi_samples),1));
opt_type = 'trust-region';
use_hess = 0;

f_min    = @(x)(-f(x));
grad_min = @(x)(-grad_f(x));

% f_min    = @(x)(f(x));
% grad_min = @(x)(grad_f(x));

if use_hess
%     hess_min = @(x)(-hess_f(x));
    hess_min = @(x)(hess_f(x));
    fun = @(x)concat3Functions(x,f_min,grad_min,hess_min);
else
    fun = @(x)concat2Functions(x,f_min,grad_min);
end

switch opt_type
    case 'trust-region'
        if use_hess
            % Optimization Options 'Trust-Region' - Conjugate Gradient (w/Hessian)
            options = optimoptions('fminunc','Display','iter','Algorithm','trust-region','SpecifyObjectiveGradient', true, 'HessianFcn','objective');
        else
            % Optimization Options 'Trust-Region' - Conjugate Gradient (w/o Hessian)
            options = optimoptions('fminunc','Display','iter','Algorithm','trust-region','SpecifyObjectiveGradient',true);
        end
    case 'quasi-newton'
        % Optimization Options 'Quasi-Newton'
        options = optimoptions('fminunc','Display','iter','Algorithm','quasi-newton','SpecifyObjectiveGradient',true);
end

% Run optimization
[x,fval,exitflag,process] = fminunc(fun,x0,options);

% Plot maxima found
h_min = scatter(x(1),x(2),70,[1 0 0],'filled');
