function [A_l, b_l, A_d, b_d] = optimize_localDS_for_LAGS(Data, A_g, att_g, ds_type, stability_vars)

% Positions and Velocity Trajectories
Xi_ref = Data(1:2,:);
Xi_ref_dot = Data(3:4,:);
[N,M] = size(Xi_ref_dot);


%%%%%%%%%%%%%% Parse Optimization Options %%%%%%%%%%%%%%
solver_type     = stability_vars.solver;

%%%%%%%%%%%%%% Initialize optimization problem with Yalmip %%%%%%%%%%%%%%
sdp_options = [];

% Define Solver to use
switch solver_type
    case 'baron'   % 'baron': Nonlinear programming solver
        sdp_options = sdpsettings('solver','baron','verbose', 1,'debug',1, 'usex0',1);
        
    case 'fmincon' % 'fmincon': Nonlinear programming solver
        sdp_options = sdpsettings('solver','fmincon','verbose', 1,'debug',1, 'usex0',1, 'allownonconvex',1, 'fmincon.algorithm','interior-point', 'fmincon.TolCon', 1e-12, 'fmincon.MaxIter', 500);
        
end
warning('off','YALMIP:strict')


%%%%%%%%%%%%%% Parse Variables for Stability Constraints %%%%%%%%%%%%%%
if stability_vars.add_constr
    constraint_type = stability_vars.constraint_type;
    chi_samples     = stability_vars.chi_samples;
    [~,M_chi]       = size(chi_samples);
    alpha           = feval(stability_vars.alpha_fun,chi_samples);
    h               = feval(stability_vars.h_fun,chi_samples);
    
    switch constraint_type
        case 'full'
            grad_lyap   = feval(stability_vars.grad_lyap_fun, chi_samples);
        case 'matrix'
            P_g = stability_vars.P_g;
            P_l = stability_vars.P_l;            
    end
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

% Define Constraints
Constraints     = [];

%%%%%%%%%%%%%% Estimate Dynamics for locally active behavior %%%%%%%%%%%%%%
if ds_type ~= 3
    
    % Variables for locally active dynamics
    sdpvar track_var
    Lambda_l  = sdpvar(N,N,'diagonal');
    b_l_var   = sdpvar(N,1);
    
    % Variables for locally deflective dynamics
    A_d_var     = sdpvar(N,N,'diagonal');
    b_d_var     = sdpvar(N,1);    
    
    % Constraints for locally active dynamics
    switch ds_type
        case 1   % 1: Symmetrically converging to ref. trajectory
            Constraints = [Lambda_l(1,1) < lambda_max_Ag Lambda_l(2,2) < lambda_max_Ag];                                                
            Constraints = [Constraints  1 <= track_var track_var <= 500]; % bounds for tracking factor
            Constraints = [Constraints  Lambda_l(2,2) < track_var*Lambda_l(1,1)];       
            Constraints = [Constraints  b_l_var == -(Q*Lambda_l*Q')*att_l];           
            
            % Assign initial values
            init_eig = -0.5; init_track = 5;
            assign(track_var, init_track);
            assign(Lambda_l(1,1), init_eig);
            assign(Lambda_l(2,2), init_eig*init_track);
            
        case 2 % 2: Symmetrically diverging from ref. trajectory
            Constraints = [Lambda_l(1,1) < 0 Lambda_l(2,2) <= -lambda_max_Ag];
            Constraints = [Constraints  Lambda_l(2,2) > track_var*abs(Lambda_l(1,1))];
            Constraints = [Constraints  200 >= track_var 1 <= track_var]; % bounds for tracking factor
            Constraints = [Constraints  b_l_var == -(Q*Lambda_l*Q')*att_l];
            
            % Assign initial values
            init_eig = -0.5; init_track = 5;
            assign(track_var, init_track);
            assign(Lambda_l(1,1), init_eig);
            assign(Lambda_l(2,2), abs(init_eig*init_track));
            
    end
    
    % Constraints for locally deflective dynamics
    if norm(att_l-att_g) < 0.5
        assign(A_d_var, -eye(N));
        assign(b_d_var, -A_d_var*att_g);
    else
        Constraints = [Constraints  A_d_var == 0.5*abs(Lambda_l(1,1))*eye(N,N) b_d_var == -A_d_var*att_l];
    end
    
    fprintf('*****Setting up optimization variables*****\n');
    % Add as constraints
    if stability_vars.add_constr
        fprintf('Adding Lyapnuov Stability Constraints...');
        total_lyap_constr_viol = sdpvar(1,1); total_lyap_constr_viol(1,1) = 0;
        for j = 1:M_chi
            % Compute local dynamics variables
            if h(j) >= 1; h_mod = 1;else; h_mod = h(j)*h_set;end
            A_L = h_mod*(Q*Lambda_l*Q') + (1-h_mod)*A_d_var;
            % Gives same results, only difference is the way the Lyapunov
            % condition is computed, either "indirectly" with the Gradient
            % of the Lyapunov function given as a function handle
            % or "directly" as defined in Eq (25)
            switch constraint_type
                case 'full' % Lyapunov Constraint using gradient of lyapnuv function
                    
                    % Compute lyapunov constraint on Chi samples
                    Constraints = [Constraints alpha(j)*grad_lyap(:,j)' * A_g * (chi_samples(:,j) - att_g) < ...
                        -(1-alpha(j))*grad_lyap(:,j)' * ((A_L) * (chi_samples(:,j) - att_l))];
                    
                    %%%%%%%%%%%% FOR DEBUGGING %%%%%%%%%%%%                    
                    chi_lyap_constr = alpha(j)*grad_lyap(:,j)' * A_g * (chi_samples(:,j) - att_g) + ...
                        (1-alpha(j))*grad_lyap(:,j)' * ((A_L) * (chi_samples(:,j) - att_l));
                    
                case 'matrix'  % Lyapunov Constraint using Q matrices                   
                    % Compute Grouped Matrices
                    Q_g  = A_g'*P_g + P_g*A_g;
                    Q_gl = A_g'*P_l;
                    Q_lg = A_L'*(2*P_g);
                    Q_l  = A_L'*P_l;                    
                    
                    % Compute Local Lyapunov Component
                    lyap_local =   (chi_samples(:,j) - att_g)'*P_l*(chi_samples(:,j) - att_l);
                    
                    % Computing activation term
                    if lyap_local >= 0
                        beta = 1;
                    else
                        beta = 0;
                    end
                    beta_l_2 = beta*2*lyap_local;
                    
                    % Computing Block Matrices
                    Q_G = alpha(j) * ( Q_g + beta_l_2*Q_gl );
                    Q_LG = (1-alpha(j))*( Q_lg + beta_l_2*Q_l );
                    Q_GL = alpha(j)*beta_l_2*Q_gl;
                    Q_L  = (1-alpha(j))*beta_l_2*Q_l;
                    Q_LGL = Q_LG + Q_GL;
                                        
                    % Compute lyapunov constraint on Chi samples
                    Constraints = [Constraints (chi_samples(:,j) - att_g)'*Q_G*(chi_samples(:,j) - att_g) < ...
                        -( (chi_samples(:,j) - att_l)'*Q_LGL*(chi_samples(:,j) - att_g)  + ...
                           (chi_samples(:,j) - att_l)'*Q_L* (chi_samples(:,j) - att_l))];
                       
                    %%%%%%%%%%%% FOR DEBUGGING %%%%%%%%%%%%
                    chi_lyap_constr = (chi_samples(:,j) - att_g)'*Q_G*(chi_samples(:,j) - att_g) + ...
                        (chi_samples(:,j) - att_l)'*Q_LGL*(chi_samples(:,j) - att_g)  + ...
                        (chi_samples(:,j) - att_l)'* Q_L *(chi_samples(:,j) - att_l);
                                                            
            end            

            % Violations
            total_lyap_constr_viol = total_lyap_constr_viol + (0.5 + 0.5*sign(chi_lyap_constr));
                        
        end
        fprintf('done \n');
    end
    
    fprintf('Computing reconstruction error...');
    %%%%%% Computing terms for the objective function %%%%%%
    % Calculate the approximated velocities with A_l
    Xi_d_dot = sdpvar(N,M, 'full');
    for m = 1:M
        % Compute the desired velocity of the reference trajectories
        Xi_d_dot(:,m) = (Q*Lambda_l*Q')*Xi_ref(:,m) + b_l_var;
    end
    % Reconstruction Error
    Xi_dot_error = Xi_d_dot - Xi_ref_dot;    
    fprintf('done \n');
    
    % Objective Formulation Type 1
    Aux_var     = sdpvar(N,length(Xi_dot_error));
    Constraints = [Constraints, Aux_var == Xi_dot_error];
    
    % Total Objective function
    Objective = (1/track_var) + 1e-6*(sum((sum(Aux_var.^2)))) ;
    

    % Solve optimization problem
    sol = optimize(Constraints, Objective, sdp_options);
    if sol.problem ~= 0
        yalmiperror(sol.problem);
    end
    
    % Optimization result
    sol.info
    fprintf('Total error: %2.2f\n', value(Objective));
    
    % Output Variables
    Lambda_l = value(Lambda_l);
    A_l = Q*Lambda_l*Q';
    b_l = value(b_l_var);
    A_d = value(A_d_var);
    b_d = value(b_d_var);

    if stability_vars.add_constr
        fprintf(2,'Total Lyapunov Constraint Violations %d \n',value(total_lyap_constr_viol));
    end
        
else
    % Estimate the local dynamics as motion pattern mimicking
    [A_l, b_l] = optimal_Ac_from_data(Data, att_l, 0);
end
