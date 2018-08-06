function [att_l, A_inv] = estimate_local_attractor(Data)

% Positions and Velocity Trajectories
Xi_ref = Data(1:2,:);
Xi_ref_dot = Data(3:4,:);
[N,M] = size(Xi_ref_dot);
min_xy = min(Xi_ref,[],2);
max_xy = max(Xi_ref,[],2);


% Solve the convex optimization problem with Yalmip
eps = 1;
sdp_options = []; Constraints = [];
warning('off','YALMIP:strict') 

% Define Variables
att_l_var  = sdpvar(N, 1);
A_inv_var  = sdpvar(N, N, 'symmetric','real');
lambda_inv_var_1  = sdpvar(1, 1);
lambda_inv_var_2  = sdpvar(1, 1);
 
% Define Constraints
sdp_options = sdpsettings('solver','penlab','verbose', 1);
Constraints = [Constraints att_l_var >= min_xy att_l_var <= max_xy] ;
% Constraints = [Constraints A_inv_var' + A_inv_var <= -eps*eye(N)];
Constraints = [Constraints lambda_inv_var_1 <= -eps lambda_inv_var_2 <= -eps];

% Calculate the ref traj. from the inverse model
Xi_dot_inv = sdpvar(N,M, 'full');
for m = 1:M
%     Xi_dot_inv(:,m) = A_inv_var*Xi_ref_dot(:,m) + att_l_var ;
    Xi_dot_inv(:,m) = [lambda_inv_var_1 0; 0 lambda_inv_var_2]*Xi_ref_dot(:,m) + att_l_var ;
end

% Then calculate the difference between approximated ref. trajectory 
% and the real one
Xi_error = Xi_ref - Xi_dot_inv;

% Defining Objective Function depending on constraints
Aux_var     = sdpvar(N,length(Xi_error));
Objective   = sum((sum(Aux_var.^2)));
Constraints = [Constraints, Aux_var == Xi_error];

% Solve optimization problem
sol = optimize(Constraints, Objective, sdp_options);
if sol.problem ~= 0
    yalmiperror(sol.problem);
end

% Optimization result
sol.info
check(Constraints)
fprintf('Total error: %2.2f\nComputation Time: %2.2f\n', value(Objective),sol.solvertime);

%%%%%% Output Variables %%%%%%
% Estimated Attractor
att_l  = value(att_l_var);

% Estimated A_inv
A_inv = value(A_inv_var);

end