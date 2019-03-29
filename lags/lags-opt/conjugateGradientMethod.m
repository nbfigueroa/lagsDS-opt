function [f_max, x_max, fvals, xvals, h_points] = conjugateGradientMethod(f,grad_f,hess_f, x0, options)

% Parse options
max_iter   = options.max_iter; % maximum number of iterations
f_tol      = options.f_tol;    % termination tolerance for F(x)
do_plot    = options.plot;     % plot init/final and iterations
be_verbose = options.verbose;   % Show values on iterations

% Gradient Ascent Iteration
iter = 1;    
x = x0;
fvals = []; xvals = [];
xvals (:,iter) = x;
fvals (iter)   = f(x);
g_k = grad_f(x);
d_k = -g_k;
converged = 0;
tic;
% Print Progress
if be_verbose
    fprintf('iter= %d  ; f(x)=%2.8f; \n',iter, fvals (iter));
end
while iter < max_iter && ~converged    
    iter = iter + 1;
    
    % Compute Hessian of current point
    H_f = hess_f(x);
    
    % Compute alpha of current iteration
    alpha_k = -(g_k'*d_k)/(d_k'*H_f*d_k);

    if isnan(alpha_k)
        break;
    end
    % Conjugate Gradient Iteration
    x = x + alpha_k * d_k;                      

    % Compute next values
    g_k     = grad_f(x);
    beta_k  = (g_k'*H_f*d_k)/(d_k'*H_f*d_k);
    d_k     = -g_k + beta_k*d_k;    
   
    
    % Check progress
    f_current = f(x);
    fdiff = fvals(iter-1) - f_current;
    % Stop if the f-difference is below a thresghold
    if (abs(fdiff) < f_tol) || (fdiff == 0)
        converged = 1;
    end
    % Stop if the x-difference is below a thresghold
    if (norm(x-xvals (:,iter-1)) < f_tol)
        converged = 1;
    end    
    
    % Print Progress    
    if be_verbose
        fprintf('iter= %d  ; f(x)=%2.8f; fdiff=%2.8f \n',iter, f_current, fdiff);
    end
    
    % Keep values
    fvals(iter)    = f_current;         
    xvals (:,iter) = x;
end
toc;

% Final values
x_max = xvals(:,end);
f_max = fvals(:,end);
fprintf('Found Maxima of function (x=%2.3f, y=%2.3f) at iter=%d with value f=%2.2f \n',x_max(1), x_max(2), iter, f_max);

if do_plot
    h_points = scatter(x_max(1),x_max(2),70,[1 0 0],'filled');
    h_points = [h_points scatter(x0(1),x0(2),40,[0 1 0],'filled')];
    h_points = [h_points plot(xvals(1,:),xvals(2,:),'-.','Color',[1 1 1],'LineWidth',2)];
else
    h_points = [];
end

end