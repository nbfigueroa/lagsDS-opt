function [att_l, A_inv] = optimize_local_attractor(Data, basis, att_l_init)

% Positions and Velocity Trajectories
Xi_ref = Data(1:2,:);
Xi_ref_dot = Data(3:4,:);
[N,M] = size(Xi_ref_dot);

% Solve the convex optimization problem with Yalmip
eps = 0.001;
sdp_options = []; Constraints = [];
warning('off','YALMIP:strict') 

inv_basis = inv(basis);

% Define Variables
att_l_var  = sdpvar(N, 1);
A_inv_var  = sdpvar(N, N, 'symmetric');
lambda_1_var  = sdpvar(1, 1);
lambda_2_var  = sdpvar(1, 1);
 
% Define Constraints
sdp_options = sdpsettings('solver','penlab','verbose', 1);
% Constraints = [Constraints A_inv_var+A_inv_var' <= -eps];
Constraints = [Constraints lambda_1_var <= -eps  lambda_2_var <= -eps];

% assign(A_inv_var,A_inv_init);
assign(att_l_var,att_l_init);

% Calculate the ref traj. from the inverse model
Xi_dot_inv = sdpvar(N,M, 'full');
for m = 1:M
%     Xi_dot_inv(:,m) = A_inv_var*Xi_ref_dot(:,m) + att_l_var ;
    Xi_dot_inv(:,m) = (inv_basis*[lambda_1_var 0; 0 lambda_2_var] *inv_basis')*Xi_ref_dot(:,m) + att_l_var ;
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
att_l  = value(att_l_var)

% Estimated A_inv
% A_inv = value(A_inv_var)
lambda1 = value(lambda_1_var)
lambda2 = value(lambda_2_var)
A_inv = inv_basis*[lambda1 0; 0 lambda2] *inv_basis'

end