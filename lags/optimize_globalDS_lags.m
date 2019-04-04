function [A_g, b_g, P] = optimize_globalDS_lags(Data, attractor, ctr_type, gmm, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2018 Learning Algorithms and Systems Laboratory,          %
% EPFL, Switzerland                                                       %
% Author:  Nadia Figueroa                                                 % 
% email:   nadia.figueroafernandez@epfl.ch                                %
% website: http://lasa.epfl.ch                                            %
%                                                                         %
% This work was supported by the EU project Cogimon H2020-ICT-23-2014.    %
%                                                                         %
% Permission is granted to copy, distribute, and/or modify this program   %
% under the terms of the GNU General Public License, version 2 or any     %
% later version published by the Free Software Foundation.                %
%                                                                         %
% This program is distributed in the hope that it will be useful, but     %
% WITHOUT ANY WARRANTY; without even the implied warranty of              %
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General%
% Public License for more details                                         %
%                                                                         %
% If you use this code in your research please cite:                      %
% "Locally Active Globally Stable DS: Theory, Learning and Experiments";  %
% N. Figueroa and A. Billard; 2019                                        %  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Positions and Velocity Trajectories
Xi_ref = Data(1:2,:);
Xi_ref_dot = Data(3:4,:);
[N,M] = size(Xi_ref_dot);

% Solve the convex optimization problem with Yalmip
sdp_options = []; Constraints = [];
epsilon = 0.01;

% Define Variables
K = length(gmm.Priors);
A_g = zeros(N,N,K);
b_g = zeros(N,K);

% Define solver for type of constraints and Initialization
switch ctr_type
    case 0
        sdp_options = sdpsettings('solver','sedumi','verbose', 1);
    
    case 1
        % 'penlab': Nonlinear semidefinite programming solver
        sdp_options = sdpsettings('solver','penlab','verbose', 1,'usex0',1);
        % Solve Problem with Convex constraints first to get A's
        fprintf('Solving Optimization Problem with Convex Constraints for Non-Convex Initialization...\n');
        [A0, b0] = optimize_lpv_ds_from_data_v2(Data, attractor, 0, gmm);
        P_var = sdpvar(N, N, 'symmetric','real');
        Constraints = [Constraints, P_var >  eye(N,N)];
        assign(P_var,eye(N));
        
    case 2
        % 'penlab': Nonlinear semidefinite programming solver
        sdp_options = sdpsettings('solver','penlab','verbose', 1,'usex0',1);
        % Solve Problem with Convex constraints first to get A's
        fprintf('Solving Optimization Problem with Convex Constraints for Non-Convex Initialization...\n');
        [A0, b0] = optimize_lpv_ds_from_data_v2(Data, attractor, 0, gmm);
        P = varargin{1};
        
    case 3
        % 'penlab': Nonlinear semidefinite programming solver
        sdp_options = sdpsettings('solver','penlab','verbose', 1,'usex0',1);
        % Solve Problem with Convex constraints first to get A's
        fprintf('Solving Optimization Problem with Convex Constraints for Non-Convex Initialization...\n');
        
        % Initialize with estimation of global DS with P_g only
        P_g = varargin{1};
        [A0, b0, ~] = optimize_lpv_ds_from_data(Data, attractor, 2, gmm, P_g, 0);                        
        
        % Parse input parametrs
        P_l = varargin{2};
        att_l = varargin{3};
        eps_scale = varargin{4};
        P_l_sum = sum(P_l,3);        
        enforce_g = varargin{5};
        if (enforce_g == 1)
            equal_g = varargin{6};
        end
end

% Posterior Probabilities per local model
h_k = posterior_probs_gmm(Xi_ref,gmm,'norm');
j = 1;
for k = 1:K    
    A_vars{k} = sdpvar(N, N, 'full','real');       
    b_vars{k} = sdpvar(N, 1, 'full');
    Q_vars{k} = sdpvar(N, N,'symmetric','real');       
                   
    switch ctr_type
        case 0 %: convex (using QLF)
            Constraints = [Constraints transpose(A_vars{k}) + A_vars{k} <= -epsilon*eye(N,N)];            
        
        case 1 %: non-convex, unknown P (using P-QLF)                                                     
            Constraints = [Constraints, transpose(A_vars{k})*P_var + P_var*A_vars{k} <= -epsilon*eye(N)];
            
            % Assign Initial Parameters
            assign(A_vars{k},A0(:,:,k));
            assign(b_vars{k},b0(:,k));
         
        case 2 %: non-convex with given P (using P-QLF)
            Constraints = [Constraints, transpose(A_vars{k})*P + P*A_vars{k} == Q_vars{k}];                        
            Constraints = [Constraints, Q_vars{k} <= -epsilon*eye(N)];                        
            
            % Assign Initial Parameters
            assign(A_vars{k},A0(:,:,k));
            assign(b_vars{k},b0(:,k));                     
            assign(Q_vars{k},-eye(N));
            
        case 3 %: non-convex with given P_g and P_l's (using WSAQF)
                        
            %%%%%% Global QLF constraint %%%%%%
            Q_g_vars{k} = sdpvar(N, N,'symmetric','real');             
            Constraints = [Constraints, transpose(A_vars{k})*P_g + P_g*A_vars{k} == Q_g_vars{k}];
            Constraints = [Constraints, Q_g_vars{k} <= -epsilon*eps_scale*eye(N)];
            
            % Assign Initial Parameters
            assign(A_vars{k},A0(:,:,k));
            assign(Q_g_vars{k},-eye(N));
                        
            %%%%%% Local QLF constraint (SUM) %%%%%%
            Q_L_vars{k} = sdpvar(N, N,'symmetric','real');
            Constraints = [Constraints, transpose(A_vars{k})*P_l_sum + P_l_sum*A_vars{k} == Q_L_vars{k}];
%             Constraints = [Constraints, transpose(A_vars{k})*P_l_sum  == Q_L_vars{k}];
            Constraints = [Constraints, Q_L_vars{k} <= -epsilon*10*eps_scale*eye(N)];           
            
            %%%%%% Local Constraints have to be more negative than global %%%%%%           
            assign(Q_L_vars{k},-eye(N));
            
            %%%%%% Constraint on the Gaussian close to the attractor %%%%%%
            if enforce_g == 1
                if k == equal_g
                    Constraints = [Constraints, transpose(A_vars{k}) + A_vars{k} <= -epsilon*eps_scale*eye(N)];
                    fprintf('If enforcing GGGG! \n');
                end
            end
            
            %%%%%% Local QLF constraint (Individual) %%%%%%
%             for kk=1:size(P_l,3)
%                 Q_l_vars{j} = sdpvar(N, N,'symmetric','real');                
%                 if beta_weights(k,kk) > 0
%                 Constraints = [Constraints, transpose(A_vars{k})*P_l(:,:,kk) + P_l(:,:,kk)*A_vars{k} == Q_l_vars{j}];
%                 Constraints = [Constraints, Q_l_vars{j} <= -epsilon*eps_scale*0.1*eye(N)];
%                 assign(Q_l_vars{j},-eye(N));
%                 j = j + 1;
%                 end
%             end          
          
    end
    
    % Define Constraints
    Constraints = [Constraints b_vars{k} == -A_vars{k}*attractor];
    assign(b_vars{k},b0(:,k));
    
end

% Calculate our estimated velocities caused by each local behavior
Xi_d_dot_c_raw = sdpvar(N,M,K, 'full');%zeros(size(Qd));
for k = 1:K
    h_K = repmat(h_k(k,:),[N 1]);
    f_k = A_vars{k}*Xi_ref + repmat(b_vars{k},[1 M]);
    Xi_d_dot_c_raw(:,:,k) = h_K.*f_k;
end

% Sum each of the local behaviors to generate the overall behavior at
% each point
Xi_d_dot = sdpvar(N, M, 'full');
Xi_d_dot = reshape(sum(Xi_d_dot_c_raw,3),[N M]);

% Then calculate the difference between approximated velocities
% and the demonstated ones for A
Xi_dot_error = Xi_d_dot - Xi_ref_dot;

% Defining Objective Function depending on constraints
if ctr_type == 0
    Xi_dot_total_error = sdpvar(1,1); Xi_dot_total_error(1,1) = 0;
    for m = 1:M
        Xi_dot_total_error = Xi_dot_total_error + norm(Xi_dot_error(:, m));
    end
    Objective = Xi_dot_total_error;
else
    Aux_var     = sdpvar(N,length(Xi_dot_error));
    Objective   = sum((sum(Aux_var.^2)));
    Constraints = [Constraints, Aux_var == Xi_dot_error];
end

% Solve optimization problem
sol = optimize(Constraints, Objective, sdp_options)
if sol.problem ~= 0
    yalmiperror(sol.problem);
end

for k = 1:K
    A_g(:,:,k) = value(A_vars{k});
    b_g(:,k)   = value(b_vars{k});
end

if exist('P_var','var')
    P = value(P_var);
else
    P = eye(N);
end

sol.info
check(Constraints)
fprintf('Total error: %2.2f\nComputation Time: %2.2f\n', value(Objective),sol.solvertime);


%%%% FOR DEBUGGING: Check Negative-Definite Constraint %%%%
if ctr_type == 3
    suff_constr_violations = zeros(1,K);
    for k=1:K
        Pg_A =  A_g(:,:,k)'*P_g + P_g*A_g(:,:,k);
        suff_constr_violations(1,k) = sum(eig(Pg_A + Pg_A') > 0); % strict
    end
    % Check Constraint Violation
    if sum(suff_constr_violations) > 0
        warning(sprintf('Global Sufficient System Matrix Constraints are NOT met..'))
    else
        fprintf('All Global Sufficient System Matrix Constraints are met..\n')
    end
    
    % Check full constraints along reference trajectory
    x_test = Xi_ref;
    full_constr_viol = zeros(1,size(x_test,2));
    gamma_k_x = posterior_probs_gmm(x_test,gmm,'norm');
    for i=1:size(x_test,2)
        A_g_k = zeros(2,2); 
        for k=1:K
            % Compute weighted A's
            A_g_k = A_g_k + gamma_k_x(k,i) * A_g(:,:,k);            
        end
        
        P_l_k = zeros(2,2);
        for j=1:size(P_l,3)
            % Compute weighted P's
            lyap_local_k =   (x_test(:,i) - attractor)'*P_l(:,:,j)*(x_test(:,i) - att_l(:,j));
            
            % Computing activation term
            if lyap_local_k >= 0
                beta = 1;
            else
                beta = 0;
            end
            beta_k_2 = 2 * beta * lyap_local_k;
            
            P_l_k = P_l_k + beta_k_2*P_l(:,:,j);
        end
        
        % Compute Q_K
        AQ = A_g_k'*(2*P_g  + P_l_k);
        full_constr_viol(1,i) = sum(eig(AQ+AQ') > 0);
    end
    % Check Constraint Violation
    if sum(full_constr_viol) > 0
        warning(sprintf('Full System Matrix Constraints are NOT met..'))
    else
        fprintf('Full System Matrix Constraints are met..\n')
    end
    
end


end