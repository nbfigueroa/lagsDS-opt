function [f_max, x_max, fvals, xvals, h_points] = newtonMethod(f,grad_f,hess_f, x0, options)

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
converged = 0;
epsilon = 1e-2;
tic;
while iter < max_iter && ~converged    
    iter = iter + 1;
    % Compute Hessian of current point
    H_f = hess_f(x);
    g_f = grad_f(x);
       
    % Eigendecomposition of Hessian
    [V, L] =  eig(H_f);   
    l_vec  = diag(L);
    if any(l_vec > 0)
        warning('Saddle point!!');
    end

    % Newton Iteration
    x = x - (H_f\g_f);                      
    
    % Check progress
    f_current = f(x);
    fdiff = fvals(iter-1) - f_current;
    if (abs(fdiff) < f_tol) || (fdiff == 0)
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