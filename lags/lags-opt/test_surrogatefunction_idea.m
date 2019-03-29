%%  Define Functions, Gradient (and Hessian?).. using surrogate f_Q (GPR)

%% Option 1:  Manually generate uniformly random distributed points in range
% Estimate point distribution
M = 2;
nb_points = 500;
[ V, D, init_mu ] = my_pca( [chi_samples att_g] );
% Do PCA on the points to project the points to
% an axis aligned embedding
[A_y, y0_all] = project_pca(chi_samples, init_mu, V, M);

% Find ranges of aligned points
Ymin_values   = min(y0_all,[],2);
Ymin_values = Ymin_values + [0.25*Ymin_values(1); 0.25*Ymin_values(2)];
Yrange_values = range(y0_all,2);
Yrange_values = Yrange_values + [0.25*Yrange_values(1); 0.25*Yrange_values(2)];

% Uniformly sample points within the ranges
init_points_y = Ymin_values(:,ones(1,nb_points)) + rand(nb_points,M)'.*(Yrange_values(:, ones(1,nb_points)));

% Project back to original manifold
chi_samples_surr = reconstruct_pca(init_points_y, A_y, init_mu);
chi_samples_surr = [chi_samples_surr att_g];

%% Option 2: Create points from chi-samples and isocontours
chi_min          = min(chi_samples,[],2);
chi_max          = max(chi_samples,[],2);
chi_edges_1      = [chi_min chi_max];
chi_edges_2      = [chi_edges_1(1,:);  chi_edges_1(2,2) chi_edges_1(2,1)];
chi_edges        = [chi_edges_1  chi_edges_2];
chi_samples_surr = [chi_samples chi_edges];

%%% Creating equidistant samples from different isocontours
desired_samples       = 10;
desired_alpha_contour = 0.99;
alpha_step = 0.05;
while desired_alpha_contour > 0
    desired_Gauss_contour = -Norm*(desired_alpha_contour-1);
    chi_samples_iso = draw_chi_samples (Sigma,Mu,desired_samples,activ_fun, 'isocontours', desired_Gauss_contour);
    chi_samples_surr = [chi_samples_surr chi_samples_iso];
    desired_alpha_contour = desired_alpha_contour - alpha_step;
end
fprintf('Using %d samples to create surrogate function \n',length(chi_samples_surr))

%% Option 3: Sample points from compact hyper-cube region of DS
nb_points = 1000; M = 2;
Xmin_values  = [limits(1); limits(3)];
Xrange_values =  [(limits(3)- limits(1)); ( limits(4) - limits(2) )];
Xrange_values = Xrange_values + [0.5*Xrange_values(1); 0.5*Xrange_values(2)];
chi_samples_surr = Xmin_values(:,ones(1,nb_points)) + rand(nb_points,M)'.*(Xrange_values(:, ones(1,nb_points)));

%% Train surrogate function of GPR
% Find optimal hyper-parameters for surrogate function (GPR)
model = [];
model.X_train = chi_samples_surr';
model.y_train = f_Q(chi_samples_surr)';
meanfunc   = {@meanZero};
covfunc    = {@covSEiso};
likfunc    = @likGauss;
hyp        = struct('mean', [], 'cov', [0 0], 'lik', -1);
hyp_opt    = minimize(hyp, @gp, -100, @infExact, meanfunc, covfunc, likfunc, model.X_train, model.y_train);

% Create Surrogate function \hat(f_Q) = E{p(f_Q|\xi)}
rbf_width = exp(hyp_opt.cov(1)); epsilon = exp(hyp_opt.lik);
f         = @(x) my_gpr(x',[],model,epsilon,rbf_width);

% Create gradient handle of surrogate function
grad_f   = @(x)gradient_gpr(x, model, epsilon, rbf_width);

% Plot surrogate function
plot_lyap_fct(f,1,limits,'Surrogate $ \hat{f}_Q(\xi)$',1);        hold on;

% Plot the gradient of the function
plot_gradient_fct(grad_f, limits,  'Surrogate $f_Q$ and $\nabla_{\xi}f_Q$ Function');

%% Plot samples used to create surrogate function
h_samples_s = scatter(chi_samples_surr(1,:),chi_samples_surr(2,:),'+','c');

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%      Optimization OPTION 1: Maxima Search using Gradient Ascent       %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Test the maxima search function
lm_options                   = [];
lm_options.type              = 'grad_ascent';
lm_options.num_ga_trials     = 10;
lm_options.do_plots          = 1;
lm_options.init_set          = chi_samples;
lm_options.verbosity         = 0;
[local_max, local_fmax]      =  find_localMaxima(f, grad_f, lm_options);
f_Q(local_max)
if any(local_fmax >= 0)
    fprintf (2, 'ALL fQ_max < 0 !!\n');
else
    fprintf ('+++++ ALL fQ_max < 0 +++++!!\n');
end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%      Optimization OPTION 2: Maxima Search using Newton Method         %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Testing newton method on surrogate function
% Plot surrogate function
plot_lyap_fct(f, 1, limits,'Surrogate $f_Q(\xi)$',1);        hold on;
h_samples_s = scatter(chi_samples_surr(1,:),chi_samples_surr(2,:),'+','c');

%%%%%%% Maxima Search Option B: Newton Method
nm_options = [];
nm_options.max_iter = 100;   % maximum number of iterations
nm_options.f_tol    = 1e-10;  % termination tolerance for F(x)
nm_options.plot     = 1;      % plot init/final and iterations
nm_options.verbose  = 1;      % Show values on iterations

% Initial value
x0 = Mu;
x0 = chi_samples(:,randsample(length(chi_samples),1));
fprintf('Finding maxima in Chi using Newton Method...\n');

% Create hessian handle of surrogate function
hess_f   = @(x)hessian_gpr(x, model, epsilon, rbf_width);

[f_max, x_max, fvals, xvals, h_points] = newtonMethod(f, grad_f, hess_f, x0, nm_options);
real_fmax = f_Q(x_max);

fprintf('Maxima of surrogate function (hat(f)_max = %2.5f, f_max = %2.5f)found at x=%2.5f,y=%2.5f \n',f_max,real_fmax,x_max(1),x_max(2));
if real_fmax > 0
    fprintf(2, 'Maxima in compact set is positive (f_max=%2.2f)! Current form is Not Stable!\n', real_fmax);
else
    fprintf('Maxima in compact set is negative(f_max=%2.2f)! Stability is ensured!\n',real_fmax);
end

