function [A_l, b_l, A_s, b_s] = estimate_localDS_max_gamma(Data, A_g, att_g, att_l, ds_type, w, P_g, P_l)

% Positions and Velocity Trajectories
Xi_ref = Data(1:2,:);
Xi_ref_dot = Data(3:4,:);
[N,M] = size(Xi_ref_dot);

% Tests for maximizing the artificial local dynamics
w_norm = -w/norm(w);
w_perp = [1;-w_norm(1)/(w_norm(2)+realmin)];

% Solve the convex optimization problem with Yalmip
sdp_options = [];
sdp_options = sdpsettings('solver','penlab','verbose', 1);
% sdp_options = sdpsettings('solver','fmincon','verbose', 1);
% sdp_options = sdpsettings('solver','mosek','verbose', 1);

% Define Constraints
Constraints     = [];
% This epsilon is super crucial for stability
epsilon = 0.1;

% Compute Unit Directions of A_l
[Q, L_, ~] =  my_pca(Xi_ref)

% Estimate Dynamics for local behavior
if ds_type ~= 3
    L_g =  eig(A_g);
    lambda_g = min(L_g);
    
    % Define Variables
    Lambda_l  = sdpvar(N, N, 'diagonal');
    A_var_l   = sdpvar(N, N, 'full','real');
    Q_var_l   = sdpvar(N, N, 'symmetric','real');
    Q_var_lg  = sdpvar(N, N, 'symmetric','real');    
    b_var     = sdpvar(N,1);
    sdpvar gamma_var;
    
    % Diverging/Converging constraints
    switch ds_type
        case 1   % 1: Symmetrically converging to ref. trajectory
            % First option, explicitly definig the directions of the A
            Constraints = [Lambda_l(1,1) <= -epsilon Lambda_l(2,2) <= -1]; 
            Constraints = [Constraints  10 >= gamma_var gamma_var <= 100]; 
            Constraints = [Constraints Lambda_l(2,2) < gamma_var*Lambda_l(1,1)];            
%             Constraints = [Constraints  A_var_l == Q*Lambda_l*Q'];
            Constraints = [Constraints  b_var   == -Q*Lambda_l*Q'*att_l];
            
            % Symmetry constraints on the system matrix
%             Constraints = [Constraints transpose(A_var_l)*(P_g) + P_g*A_var_l == Q_var_lg];
%             Constraints = [Constraints, Q_var_lg < -eye(N)];

            % Negative eigenvalues for symmetric part
%             Constraints = [Constraints 0.5*(transpose(A_var_l)*P_l + transpose(transpose(A_var_l)*P_l)) < -eye(N)];           
            
            % Enforce symmertry on local component
%             Constraints = [Constraints transpose(A_var_l)*P_l == Q_var_l];
%             Constraints = [Constraints, Q_var_l < -eye(N)];


        case 2 % 2: Symmetrically diverging from ref. trajectory
            Constraints = [Lambda_l(1,1) < 0 Lambda_l(2,2) > 0  Lambda_l(1,1) <= (real(lambda_g) + imag(lambda_g))];
            Constraints = [Constraints  Lambda_l(2,2) > tracking_factor*abs(Lambda_l(1,1))];            
            
            % Symmetry constraints on the system matrix
            Constraints = [Constraints Q*Lambda_l*Q' ==  A_var_l];
            Constraints = [Constraints  b_var == -A_var_l*att_l];
            
            Constraints = [Constraints transpose(Q*Lambda_l*Q')*P_g' + transpose(Q*Lambda_l*Q')*P_g == Q_var_lg];
            Constraints = [Constraints transpose(Q*Lambda_l*Q')*P_l == Q_var_l];
            
            Constraints = [Constraints, Q_var_lg >= epsilon*eye(N)];
            Constraints = [Constraints, Q_var_l >= epsilon*eye(N)];
            
            assign(Q_var_l,10*eye(N));
            assign(Q_var_lg,10*eye(N));
            
    end
    
    % Calculate the approximated velocities with A_c
    Xi_d_dot_t = sdpvar(N,M, 'full');
    
    % Then calculate the difference between approximated velocities
    % and the demonstrated ones for A_t
    for m = 1:M
            Xi_d_dot_t(:,m) = (Q*Lambda_l*Q')*Xi_ref(:,m) + b_var;
%             Xi_d_dot_t(:,m) = A_var_l*Xi_ref(:,m) + b_var;
    end
    
    % Then calculate the difference between approximated velocities
    % and the demonstated ones with A_t
    Xi_dot_error_t = Xi_d_dot_t - Xi_ref_dot;
    Aux_var     = sdpvar(N,length(Xi_dot_error_t));    
    Constraints = [Constraints, Aux_var == Xi_dot_error_t];
%     Objective = 0.001*sum((sum(Aux_var.^2)));
    Objective   = sum((sum(Aux_var.^2)));
    
    % Solve optimization problem
    sol = optimize(Constraints,Objective,sdpsettings);
    if sol.problem ~= 0
        yalmiperror(sol.problem);
    end
    
    % Output Variables
    value(gamma_var)
    Lambda_l = value(Lambda_l)
    A_l = Q*Lambda_l*Q'
%     A_l = value(A_var_l)
    check(Constraints)
    fprintf('Total error: %2.2f\n', value(Objective));
else
    % Estimate the local dynamics as motion pattern mimicking
    [A_l, ~] = optimal_Ac_from_data(Data, att_l, 0);
end
b_l = [0 0]';


%%%%%%%%%%%%%% Estimate Dynamics for locally deflective behavior %%%%%%%%%%%%%%
%%% Stuff for deflective DS -- Compute angle between local and global DS %%%
att_vec     = att_g - att_l;
angle_w  = atan2(w_norm(2),w_norm(1))-atan2(w_perp(2),w_perp(1));
if angle_w < 0
    w_perp = -w_perp./norm(w_perp);
else
    w_perp = w_perp./norm(w_perp);
end
angle  = atan2(att_vec(2),att_vec(1))-atan2(w_perp(2),w_perp(1));

% put in a good range
if(angle > pi)
    angle = -(2*pi-angle);
elseif(angle < -pi)
    angle = 2*pi+angle;
end
theta = angle;

% For Local Dynamics Stabilization
[Q_l, Lambda_l] = eig(A_l);
% lambda_s = (max(abs(diag(Lambda_l))) + min(abs(diag(Lambda_l))))/2;
lambda_s = (0.5)*min(abs(diag(Lambda_l)));
A_s =  lambda_s * eye(2);

if norm(att_l-att_g) < 0.5
    A_s = zeros(2);
end
% A_s = eye(1);

if (theta < -pi/2*0.8) && (theta > -pi/2*1.2)
    b_s  = -A_s*[att_l];
    A_s = Q_l*[-min(diag(Lambda_l)) 0; 0 -10*min(diag(Lambda_l))]*Q_l';
else              
    b_s = -A_s*att_l;
end

end