function [A_l, b_l, A_d, b_d] = optimize_localDS_for_LAGS_analytic(Data, A_g, att_g, ds_type, w, stability_vars)

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
epsilon = 0.01;

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

% For Non-linear problems
sdp_options = sdpsettings('solver','fmincon','verbose', 1,'debug',1, 'usex0',1, 'allownonconvex',1, 'fmincon.algorithm','interior-point', 'fmincon.TolCon', 1e-12, 'fmincon.MaxIter', 500);    
% sdp_options = sdpsettings('solver','baron','verbose', 1,'debug', 1, 'usex0',1); 
warning('off','YALMIP:strict') 

% Define Constraints
Constraints     = [];

% predefine A_d
A_d = eye(N); b_d = -A_d*att_l;

Lambda_Ag = eig(A_g);
lambda_max_Ag = max(Lambda_Ag);

%%%%%%%%%%%%%% Estimate Dynamics for locally active behavior %%%%%%%%%%%%%%
if ds_type ~= 3
    
    % Variables for locally active dynamics
    sdpvar track_var
    Lambda_l  = sdpvar(N,N,'diagonal');
    A_var_l   = sdpvar(N, N, 'full');
    b_l_var   = sdpvar(N,1);
    Q_var_lg  = sdpvar(N, N, 'full');        
    
    % Variables for locally deflective dynamics
    A_d_var     = sdpvar(N,N,'diagonal');
    b_d_var     = sdpvar(N,1);    
    
    % Variables from Stability Constraints
    P_g = stability_vars.P_g;
    P_l = stability_vars.P_l;
    sdpvar lambda_1_Q_lg lambda_2_Q_lg
        
    % Constraints for locally active dynamics
    switch ds_type
        case 1   % 1: Symmetrically converging to ref. trajectory
           Constraints = [Lambda_l(1,1) <= lambda_max_Ag Lambda_l(2,2) <= lambda_max_Ag];                                           
            Constraints = [Constraints  1 <= track_var track_var <= 200]; % bounds for tracking factor
            Constraints = [Constraints  Lambda_l(2,2) < track_var*Lambda_l(1,1)];       
            Constraints = [Constraints  A_var_l == Q*Lambda_l*Q'];
            Constraints = [Constraints  b_l_var == -A_var_l*att_l];           
            
            % Add stability constraints
            if stability_vars.add_constr
                Constraints = [Constraints 2*transpose(A_var_l)*P_g  == Q_var_lg];
                lambda_1_Q_lg =   0.5*( (Q_var_lg(1,1) + Q_var_lg(2,2)) - sqrtm( (Q_var_lg(1,1) - Q_var_lg(2,2))^2 + 4*Q_var_lg(1,2)));
%                 lambda_2_Q_lg =   0.5*( (Q_var_lg(1,1) + Q_var_lg(2,2)) + sqrtm( (Q_var_lg(1,1) - Q_var_lg(2,2))^2 + 4*Q_var_lg(1,2)));
                Constraints = [Constraints lambda_1_Q_lg < -epsilon];
%                 Constraints = [Constraints lambda_2_Q_lg <= -epsilon];
            end

            
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
%             Xi_d_dot(:,m) = (Q*Lambda_l*Q')*Xi_ref(:,m) + b_l_var;
            Xi_d_dot(:,m) = A_var_l*Xi_ref(:,m) + b_l_var;
    end
    % Reconstruction Error
    Xi_dot_error = Xi_d_dot - Xi_ref_dot;    
    fprintf('done \n');
    
    % Objective Formulation Type 1
    Aux_var     = sdpvar(N,length(Xi_dot_error));    
    Constraints = [Constraints, Aux_var == Xi_dot_error];   
    
    % Total Objective function
    Objective = (1/track_var) + 1e-5*(sum((sum(Aux_var.^2)))) ;
%     Objective = sum((sum(Aux_var.^2))) ;
    
    
    % Solve optimization problem
    sol = optimize(Constraints, Objective, sdp_options)
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
        Q_lg = value(Q_var_lg)
        eig(Q_lg)
%         Q_l = value(Q_var_l)
%         eig(Q_l)
    end
else
    % Estimate the local dynamics as motion pattern mimicking
    [A_l, b_l] = optimal_Ac_from_data(Data, att_l, 0);
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