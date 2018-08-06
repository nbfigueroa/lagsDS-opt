function [A_l, b_l, A_d, b_d] = optimize_localDS_for_LAGS(Data, A_g, att_g, ds_type, w, stability_vars)

% Positions and Velocity Trajectories
Xi_ref = Data(1:2,:);
Xi_ref_dot = Data(3:4,:);
[N,M] = size(Xi_ref_dot);

% Attractor and Directions (Pre-computed geometrically)
% Compute Unit Directions of A_l
[Q, L, ~] =  my_pca(Xi_ref);
att_l = Xi_ref(:,end);

% Tests for maximizing the artificial local dynamics
% w_norm = -w/norm(w);
% w_perp = [1;-w_norm(1)/(w_norm(2)+realmin)];
% Xi_ref_eps = Xi_ref(:,1:2:end) + w_perp*0.25;
% M_art = size(Xi_ref_eps,2);

% Optimization variables
epsilon = 0.0001;

% Solve the optimization problem with Yalmip
sdp_options = [];

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
if angle_n > pi/2 || angle_n < -pi/2
    h_set = 0;
    corr_scale = 5;
else
    h_set = 1;
    corr_scale = 1;
end

%%%%%%%%%%%%%% Estimate Values for Stability Constraints %%%%%%%%%%%%%%
chi_samples = stability_vars.chi_samples;
[~,M_chi]   = size(chi_samples);
grad_lyap   = feval(stability_vars.grad_lyap_fun, chi_samples);
alpha       = feval(stability_vars.alpha_fun,chi_samples);
h           = feval(stability_vars.h_fun,chi_samples);
lambda      = feval(stability_vars.lambda_fun,chi_samples);
grad_h      = feval(stability_vars.grad_h_fun,chi_samples);

% For Non-linear problems
sdp_options = sdpsettings('solver','fmincon','verbose', 1,'debug',1, 'usex0',1, 'allownonconvex',1, 'fmincon.algorithm','interior-point', 'fmincon.TolCon', 1e-12);    
warning('off','YALMIP:strict') 

% Define Constraints
Constraints     = [];

% predefine A_d
A_d = eye(N); b_d = -A_d*att_l;

%%%%%%%%%%%%%% Estimate Dynamics for locally active behavior %%%%%%%%%%%%%%
if ds_type ~= 3
    
    % Variables for locally active dynamics
    sdpvar track_var
    Lambda_l  = sdpvar(N,N,'diagonal');
    b_l_var     = sdpvar(N,1);
    
    % Variables for locally deflective dynamics
    A_d_var     = sdpvar(N,N,'diagonal');
    b_d_var     = sdpvar(N,1);    
    
    % Constraints for locally active dynamics
    switch ds_type
        case 1   % 1: Symmetrically converging to ref. trajectory
            Constraints = [Lambda_l(1,1) <= -1 Lambda_l(2,2) <= -1];                                                
            Constraints = [Constraints  1 <= track_var  track_var <= 100 ]; % bounds for tracking factor
            Constraints = [Constraints  Lambda_l(2,2) < track_var*Lambda_l(1,1)];       
            Constraints = [Constraints  b_l_var == -(Q*Lambda_l*Q')*att_l];
            
            % Assign initial values
            init_eig = -0.5; init_track = 5;
            assign(track_var, init_track);
            assign(Lambda_l(1,1), init_eig);
            assign(Lambda_l(2,2), init_eig*init_track);
            
        case 2 % 2: Symmetrically diverging from ref. trajectory
            Constraints = [Lambda_l(1,1) <= -1 Lambda_l(2,2) >= 1];
            Constraints = [Constraints  Lambda_l(2,2) > track_var*abs(Lambda_l(1,1))];
            Constraints = [Constraints  100 >= track_var 1 <= track_var]; % bounds for tracking factor
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
    fprintf('Computing reconstruction error...');
    % For first term of the objective function
    % Calculate the approximated velocities with A_l
    Xi_d_dot = sdpvar(N,M, 'full');
    for m = 1:M
            % Compute the desired velocity of the reference trajectories
            Xi_d_dot(:,m) = (Q*Lambda_l*Q')*Xi_ref(:,m) + b_l_var;
    end
         
    %%%%%% Computing terms for the objective function %%%%%%
    % Reconstruction Error
    Xi_dot_error = Xi_d_dot - Xi_ref_dot;    
    Xi_dot_total_error = sdpvar(1,1); Xi_dot_total_error(1,1) = 0;    
    for i = 1:M
        Xi_dot_total_error = Xi_dot_total_error +  norm(Xi_dot_error(:, i));
    end
    fprintf('done \n');
    
    % Add as constraints
    if stability_vars.add_constr
        fprintf('Computing Lyapnuov Stability Constraints...');
        total_lyap_constr_viol = sdpvar(1,1); total_lyap_constr_viol(1,1) = 0;
        for j = 1:M_chi
            % Compute local dynamics variables
            if h(j) >= 1; h_mod = 1;else; h_mod = h(j)*h_set;end
            A_L = h_mod*(Q*Lambda_l*Q') + (1-h_mod)*A_d_var;
            
            % Compute lyapunov constraint on Chi samples
            chi_lyap_constr = alpha(j)*grad_lyap(:,j)' * A_g * (chi_samples(:,j) - att_g) + ...
                (1-alpha(i))*grad_lyap(:,j)' * ((A_L) * (chi_samples(:,j) - att_l) - corr_scale*lambda(j)* grad_h(:,j));
            
            % Violations
            total_lyap_constr_viol = total_lyap_constr_viol + (0.5 + 0.5*sign(chi_lyap_constr));
            Constraints = [Constraints alpha(j)*grad_lyap(:,j)' * A_g * (chi_samples(:,j) - att_g) < -(1-alpha(i))*grad_lyap(:,j)' * ((A_L) * (chi_samples(:,j) - att_l) - corr_scale*lambda(j)* grad_h(:,j))];
        end
        fprintf('done \n');
    end
    
    
    % Total Objective function
    Objective = (1/track_var) + 1/M*Xi_dot_total_error ;
    
    % Solve optimization problem
    sol = optimize(Constraints, Objective, sdp_options);
    if sol.problem ~= 0
        yalmiperror(sol.problem);
    end
    
    % Optimization result
    sol.info
%     check(Constraints)
    fprintf('Total error: %2.2f\n', value(Objective));
    
    % Output Variables
    Lambda_l = value(Lambda_l);
    A_l = Q*Lambda_l*Q';
    b_l = value(b_l_var);
    A_d = value(A_d_var);
    b_d = value(b_d_var);
    
    if stability_vars.add_constr
        total_viol = value(total_lyap_constr_viol)        
    end
    
    
else
    % Estimate the local dynamics as motion pattern mimicking
    [A_l, b_l] = optimal_Ac_from_data(Data, att_l, 0);
end


% Check for stability
lyap_constr = zeros(1,M_chi);
for j = 1:M_chi
   
    % Compute local dynamics variables
    if h(j) >= 1
        h_mod = 1;
    else
        h_mod = h(j)*h_set;
    end
    A_L = h_mod*A_l + (1-h_mod)*A_d;
    
    
    % Computing full derivative
    lyap_constr(1,j) = alpha(j)*grad_lyap(:,j)' * (A_g) * (chi_samples(:,j) - att_g) + ...
                       (1-alpha(j))*grad_lyap(:,j)' * ((A_L) * (chi_samples(:,j) - att_l) - corr_scale*lambda(j)* grad_h(:,j));
    
end

% Violations of Necessary Constraints
violations       = lyap_constr >= 0;

% Check Constraint Violation
if sum(violations) > 0 
    warning(sprintf('System is not stable.. %d Necessary (grad) Lyapunov Violations found', sum(violations)))
else
    fprintf('System is stable..')
end

end

%%%%%%%%%%%%%% Estimate Dynamics for locally deflective behavior %%%%%%%%%%%%%%
%%% Stuff for deflective DS -- Compute angle between local and global DS %%%
% att_vec     = att_g - att_l;
% angle_w  = atan2(w_norm(2),w_norm(1))-atan2(w_perp(2),w_perp(1));
% if angle_w < 0
%     w_perp = -w_perp./norm(w_perp);
% else
%     w_perp = w_perp./norm(w_perp);
% end
% angle  = atan2(att_vec(2),att_vec(1))-atan2(w_perp(2),w_perp(1));
% 
% % put in a good range
% if(angle > pi)
%     angle = -(2*pi-angle);
% elseif(angle < -pi)
%     angle = 2*pi+angle;
% end
% theta = angle;

% if (theta < -pi/2*0.8) && (theta > -pi/2*1.2)
%     A_s = Q_l*[-min(diag(Lambda_l)) 0; 0 -10*min(diag(Lambda_l))]*Q_l';
% end