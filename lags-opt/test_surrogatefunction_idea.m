%%  Define Functions, Gradient (and Hessian?).. using surrogate f_Q (GPR)
% Create equidistant (and edge) samples for surrogate function
% Creating edge samples
chi_min          = min(chi_samples,[],2);
chi_max          = max(chi_samples,[],2);
% limits           = [chi_min(1) chi_max(1) chi_min(2) chi_max(2)];
chi_edges_1      = [chi_min chi_max];
chi_edges_2      = [chi_edges_1(1,:);  chi_edges_1(2,2) chi_edges_1(2,1)];
chi_edges        = [chi_edges_1  chi_edges_2];
chi_samples_surr = [chi_samples_used chi_edges chi_samples_test(:,randsample(length(chi_samples_test),100))];

% Creating equidistant samples from different isocontours
desired_samples       = 10;
desired_alpha_contour = 0.95;
alpha_step = 0.15;
while desired_alpha_contour > 0
    desired_Gauss_contour = -Norm*(desired_alpha_contour-1);
    chi_samples_iso = draw_chi_samples (Sigma,Mu,desired_samples,activ_fun, 'isocontours', desired_Gauss_contour);
    chi_samples_surr = [chi_samples_surr chi_samples_iso];
    desired_alpha_contour = desired_alpha_contour - alpha_step;
end
fprintf('Using %d samples to create surrogate function \n',length(chi_samples_surr))

% Find optimal hyper-parameters for surrogate function (GPR)
model = [];
model.X_train = chi_samples_surr';
model.y_train = f_Q(chi_samples_surr)';
meanfunc   = {@meanZero};
covfunc    = {@covSEiso};
likfunc    = @likGauss;
hyp        = struct('mean', [], 'cov', [0 0], 'lik', -1);
hyp_opt    = minimize(hyp, @gp, -100, @infLaplace, meanfunc, covfunc, likfunc, model.X_train, model.y_train);

% Create Surrogate function \hat(f_Q) = E{p(f_Q|\xi)}
rbf_width = exp(hyp_opt.cov(1)); epsilon = 0.001;
f         = @(x) my_gpr(x',[],model,epsilon,rbf_width);

%%%%% Implement Newton Method to find maxima in compact set (on Surrogate function!)  %%%%%
% Create gradient handle of surrogate function
grad_f   = @(x)gradient_gpr(x, model, epsilon, rbf_width);

% Plot surrogate function
plot_lyap_fct(f,1,limits_,'$f_Q(\xi)$',1);        hold on;
h_samples_s = scatter(chi_samples_surr(1,:),chi_samples_surr(2,:),'+','c');

% Plot the gradient of the function
eval_fun       = @(x)alpha_fun(x);
plot_gradient_fct(grad_f, limits_,  '$f_Q$ and $\nabla_{\xi}f_Q$ Function');

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%      Optimization OPTION 1: Maxima Search using Gradient Ascent       %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ga_options = [];
ga_options.gamma    = 0.0001;  % step size (learning rate)
ga_options.max_iter = 10000;    % maximum number of iterations
ga_options.f_tol    = 1e-10;   % termination tolerance for F(x)
ga_options.plot     = 1;       % plot init/final and iterations
ga_options.verbose  = 1;       % Show values on iterations

% Set Initial value
x0 = Mu;
inner_sample = 0;
while ~inner_sample    
%     x0 = Xi_ref(:,randsample(length(Xi_ref),1));    
    x0 = chi_samples_used(:,randsample(length(chi_samples_used),1));
    x0 = att_l;
    if lambda_fun(x0) < 0.5 && alpha_fun(x0) < 0.5
        inner_sample = 1;
    end
end
fprintf('Finding maxima in Test function using Gradient Ascent...\n');

%% Plot the Lyapunov function to find minima
% plot_lyap_fct(f_Q, 0, limits_,'Test $f(\xi)$',1);
% x0 = att_l;
x0 = chi_samples_used(:,randsample(length(chi_samples_used),1));
[f_max, x_max, fvals, xvals, h_points] = gradientAscent(f_Q,grad_fQ, x0, ga_options);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%      Optimization OPTION 2: Maxima Search using Newton Method         %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% CURRENTLY ONLY WORKS FOR THE FIRST FUNCTION TYPE!!!
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


%% Testing newton method on surrogate function
% Plot surrogate function
plot_lyap_fct(surr_fQ, 1, limits,'Surrogate $f_Q(\xi)$',1);        hold on;
h_samples_s = scatter(chi_samples_surr(1,:),chi_samples_surr(2,:),'+','c');

%%%%%%% Maxima Search Option B: Newton Method
nm_options = [];
nm_options.max_iter = 2;   % maximum number of iterations
nm_options.f_tol    = 1e-10;  % termination tolerance for F(x)
nm_options.plot     = 1;      % plot init/final and iterations
nm_options.verbose  = 1;      % Show values on iterations

% Initial value
x0 = Mu;
% x0 = chi_samples_used(:,50+randsample(2*desired_samples,1));
fprintf('Finding maxima in Chi using Newton Method...\n');

% Create hessian handle of surrogate function
hess_surr_fQ   = @(x)hessian_gpr(x, model, epsilon, rbf_width);

[f_max, x_max, fvals, xvals, h_points] = newtonMethod(surr_fQ, grad_surr_fQ, hess_surr_fQ, x0, nm_options);
real_fmax = f_Q(x_max);

fprintf('Maxima of surrogate function (hat(f)_max = %2.5f, f_max = %2.5f)found at x=%2.5f,y=%2.5f \n',f_max,real_fmax,x_max(1),x_max(2));
if real_fmax > 0
    fprintf(2, 'Maxima in compact set is positive (f_max=%2.2f)! Current form is Not Stable!\n', real_fmax);
else
    fprintf('Maxima in compact set is negative(f_max=%2.2f)! Stability is ensured!\n',real_fmax);
end

