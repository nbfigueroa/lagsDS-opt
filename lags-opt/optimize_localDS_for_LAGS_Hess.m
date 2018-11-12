function [A_l, b_l, A_d, b_d] = optimize_localDS_for_LAGS_Hess(Data, A_g, att_g, stability_vars)

% Positions and Velocity Trajectories
Xi_ref = Data(1:2,:);
Xi_ref_dot = Data(3:4,:);
[N,M] = size(Xi_ref_dot);

%%%%%%%%%%%%%% Parse Optimization Options %%%%%%%%%%%%%%
% solver_type     = stability_vars.solver;
epsilon = 1e-6;

%%%%%%%%%%%%%% Parse Variables for Stability Constraints %%%%%%%%%%%%%%
if stability_vars.add_constr
    chi_samples     = stability_vars.chi_samples;
    alpha           = feval(stability_vars.alpha_fun,chi_samples);
    h               = feval(stability_vars.h_fun,chi_samples);    
    P_g = stability_vars.P_g;
    P_l = stability_vars.P_l;
end

%%%%%%%%%%%%%% Pre-compute/Extract Variables for Optimization %%%%%%%%%%%%%%
% Attractor and Directions (Pre-computed geometrically)
[Q, ~, ~] =  my_pca(Xi_ref);
att_l = Xi_ref(:,end);

% Check incidence angle at local attractor
w = stability_vars.grad_h_fun(att_l);
w_norm = -w/norm(w);
fg_att = A_g*att_l;
fg_att_norm = fg_att/norm(fg_att);

% Put angles in nice range
angle_n  = atan2(fg_att_norm(2),fg_att_norm(1)) - atan2(w_norm(2),w_norm(1));
if(angle_n > pi)
    angle_n = -(2*pi-angle_n);
elseif(angle_n < -pi)
    angle_n = 2*pi+angle_n;
end

% Check if it's going against the grain
% if angle_n > pi/2 || angle_n < -pi/2
%     h_set = 0;
%     corr_scale = 5;
% else
%     h_set = 1;
%     corr_scale = 1;
% end

% predefine A_d
A_d = eye(N); b_d = -A_d*att_l;
Lambda_Ag = eig(A_g);
lambda_max_Ag = max(Lambda_Ag);

% Define Optimization Functions
p0  = shape_opt_params (lambda_max_Ag*eye(N), 1);
f   = @(p)objective_function(p, Xi_ref, Xi_ref_dot, Q, att_l, epsilon);
c1  = @(p)constraints_function(p, lambda_max_Ag, N);

if stability_vars.add_constr
    c2  = @(p)constraints_function_hess(p, att_l, A_g, att_g, Q, P_g, P_l, alpha, h, A_d, chi_samples, epsilon, h_set);
    c =  @(x) Constraint(x, c1, c2);
else
    c =  @(x) Constraint(x, c1);
end
    
% Optimization options
options.tol_stopping=10^-10; 
options.max_iter = 500;
options.display = 1;

% Running the optimization
if options.display
    str = 'iter';
else
    str = 'off';
end

% Options for NLP Solvers
optNLP = optimset( 'Algorithm', 'interior-point', 'LargeScale', 'off',...
    'GradObj', 'off', 'GradConstr', 'off', 'DerivativeCheck', 'on', ...
    'Display', 'iter', 'TolX', options.tol_stopping, 'TolFun', options.tol_stopping, 'TolCon', 1e-12, ...
    'MaxFunEval', 200000, 'MaxIter', options.max_iter, 'DiffMinChange', ...
    1e-4, 'Hessian','off','display',str);


% Solve the NLP Problem
[p_opt J] = fmincon(f, p0, [],[],[],[],[],[], c, optNLP);

[Lambda_l, gamma] = reshape_opt_params (p_opt);
A_l = Q*Lambda_l*Q';
b_l = -A_l*att_l;

end
    
function [J, dJ] = objective_function(p, xi_ref, xi_ref_dot, Q, att_l, epsilon)
    [Lambda_l, gamma] = reshape_opt_params (p);
    [M,N] = size(xi_ref);
    xi_d_dot = zeros(size(xi_ref_dot));
    for m = 1:N
        % Compute the desired velocity of the reference trajectories
        xi_d_dot(:,m) = (Q*Lambda_l*Q')*( xi_ref(:,m) - att_l);
    end
    
    % Reconstruction Error
    xi_dot_error = sum(vecnorm(xi_ref_dot-xi_d_dot));
    
    % Objective Function
    J = 1/gamma + epsilon*xi_dot_error;   
    dJ = [];
end

function [p] = shape_opt_params (Lambda_l, gamma)
    p = diag(Lambda_l);
    p = [p; gamma];
end

function [Lambda_l, gamma] = reshape_opt_params (p)
    Lambda_l = diag(p(1:end-1));
    gamma = p(end);    
end

function [c,ceq,dc,dceq] =  constraints_function(p, lambda_max_Ag, N)
% Constraints on System matrix (Make variable with N)
c = [];
for n=1:N
    c = [c; p(n) - lambda_max_Ag];
end
c = [c; -(p(N+1)-1)];
c = [c; p(N+1)-500];

% This should change when N>2
c = [c; p(N) - p(N+1)*p(N-1)];

% Remaining Output Variables
ceq  = [];
dc   = [];
dceq = [];
end

function [c,ceq,dc,dceq] = constraints_function_hess(p, att_l, A_g, att_g, Q,  P_g, P_l, alpha, h, A_d, chi_samples, epsilon, h_set)

% Compute local dynamics variables
[Lambda_l, ~] = reshape_opt_params (p);

% Compute Grouped "Global" Matrices
Q_g  = A_g'*P_g + P_g*A_g;
Q_gl = A_g'*P_l;

c = [];
[~,M_chi]       = size(chi_samples);
for m=1:M_chi
    if h(m) >= 1; h_mod = 1;else; h_mod = h(m)*h_set;end
    A_L = h_mod*(Q*Lambda_l*Q') + (1-h_mod)*A_d;
    
    % Compute Grouped "Local" Matrices
    Q_lg = A_L'*(2*P_g);
    Q_l  = A_L'*P_l;
    
    % Compute Local Lyapunov Component
    lyap_local =   (chi_samples(:,m) - att_g)'*P_l*(chi_samples(:,m) - att_l);
    
    % Computing activation term
    if lyap_local >= 0
        beta = 1;
    else
        beta = 0;
    end
    beta_l_2 = beta*2*lyap_local;
    
    % Computing Block Matrices
    Q_G = alpha(m) * ( Q_g + beta_l_2*Q_gl );
    Q_LG = (1-alpha(m))*( Q_lg + beta_l_2*Q_l );
    Q_GL = alpha(m)*beta_l_2*Q_gl;
    Q_L  = (1-alpha(m))*beta_l_2*Q_l;
    Q_LGL = Q_LG + Q_GL;
    
    % Symmetric form of Big Q (Compute analytically)       
    Big_Q_sym = [Q_G 0.5*Q_LGL'; 0.5*Q_LGL 0.5*(Q_L+Q_L')];    
        
    % Hessian of Current Lyapunov-constraint function f_Q
    A = Big_Q_sym(1:2,1:2);   B = Big_Q_sym(1:2,3:4); 
    B_T = Big_Q_sym(3:4,1:2); C = Big_Q_sym(3:4,3:4);   
    H_fQ = 2*A + 2*(B+B_T) + 2*C;
    
    % Eigenvalues of Hessian
    eigs_HfQ = eig(H_fQ);    
    
    c = [c;  eigs_HfQ(1) - epsilon]; 
    c = [c;  eigs_HfQ(2) - epsilon]; 
%     if eigs_HfQ(2) < 0
%         c = [c;  eigs_HfQ(1) - epsilon]; 
%     elseif eigs_HfQ(2) > 0
%         c = [c;  -eigs_HfQ(1) + epsilon]; 
%     else
%         c = [c; inf];
%     end
%     c = [c;  -( det(A) + det(B + B_T + C))]; 

end

% Remaining Output Variables
ceq  = [];
dc   =  [];
dceq = [];
end

