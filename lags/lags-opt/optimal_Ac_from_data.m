function [A_c, b_c, P] = optimal_Ac_from_data(Data, attractor, ctr_type, varargin)

% Positions and Velocity Trajectories
Xi_ref = Data(1:2,:);
Xi_ref_dot = Data(3:4,:);
[N,M] = size(Xi_ref_dot);

% Solve the convex optimization problem with Yalmip
sdp_options = []; Constraints = [];
epsilon = 0.0001;

if nargin == 3
    
    % Define Variables
    A_var = sdpvar(N, N, 'full','real');
%     A_var = sdpvar(N, N, 'symmetric','real');
    b_var = sdpvar(N, 1);
    
    % Define Constraints        
    switch ctr_type
        case 0
            sdp_options = sdpsettings('solver','sedumi','verbose', 1);
            Constraints = [Constraints A_var' + A_var <= -epsilon*eye(N,N) b_var == -A_var*attractor ];
            
        case 1
            % 'penlab': Nonlinear semidefinite programming solver
            sdp_options = sdpsettings('solver','penlab','verbose', 1,'usex0',1);
            % Solve Problem with Convex constraints first to get A's
            fprintf('Solving Optimization Problem with Convex Constraints for Non-Convex Initialization...\n');
            [A0, ~] = optimal_Ac_from_data(Data, attractor, 0);
            assign(A_var,A0);
            P_var = sdpvar(N, N,'symmetric');
            Constraints = [Constraints, (A_var'*P_var + P_var*A_var) <= -epsilon*eye(N)];            
            Constraints = [Constraints,  P_var > epsilon*eye(N), b_var == -A_var*attractor];                                    
    end
    
    % Calculate the approximated velocities with A_c
    Xi_d_dot = sdpvar(N,M, 'full');
    for m = 1:M
        Xi_d_dot(:,m) = A_var*Xi_ref(:,m) + b_var;
    end
    
    % Then calculate the difference between approximated velocities
    % and the demonstated ones for A_c
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
    sol = optimize(Constraints, Objective, sdp_options);
    if sol.problem ~= 0
        yalmiperror(sol.problem);
    end
    A_c = value(A_var);   
    b_c = value(b_var);
    if exist('P_var','var')
        P = value(P_var);
    else
        P = eye(N);
    end
else 
    
    % Define Variables
    gmm = varargin{1};
    K = length(gmm.Priors);
    A_c = zeros(N,N,K);
    b_c = zeros(N,K);    
    
    
    % Define solver for type of constraints and Initialization (for NLP)
    switch ctr_type
        case 0
            sdp_options = sdpsettings('solver','sedumi','verbose', 1);
        case 1
            % 'penlab': Nonlinear semidefinite programming solver
            sdp_options = sdpsettings('solver','penlab','verbose', 1,'usex0',1);           
            % Solve Problem with Convex constraints first to get A's
            fprintf('Solving Optimization Problem with Convex Constraints for Non-Convex Initialization...\n');
            [A0, b0] = optimal_Ac_from_data(Data, attractor, 0, gmm);
        case 2
            % 'penlab': Nonlinear semidefinite programming solver
            sdp_options = sdpsettings('solver','penlab','verbose', 1,'usex0',1);
            % Solve Problem with Convex constraints first to get A's
            fprintf('Solving Optimization Problem with Convex Constraints for Non-Convex Initialization...\n');
            [A0, b0] = optimal_Ac_from_data(Data, attractor, 0, gmm);           
    end
  
    
    % Define Constraints            
    if ctr_type == 1
        P_var = sdpvar(N, N,'symmetric','real');
        Constraints = [Constraints, P_var >= epsilon*eye(N,N)];
        assign(P_var,eye(N));
    end
    
    for k = 1:K
        A_vars{k} = sdpvar(N, N, 'full');
%         A_vars{k} = sdpvar(N, N, 'symmetric');
        b_vars{k} = sdpvar(N, 1, 'full');
            % Define Constraints        
            switch ctr_type
                case 0 %: convex
                    Constraints = [Constraints A_vars{k}'+A_vars{k} <= -epsilon*eye(N,N), b_vars{k} == -A_vars{k}*attractor];
                case 1 %: non-convex                                                            
                    Constraints = [Constraints, (A_vars{k}'*P_var + P_var*A_vars{k}) <= -epsilon*eye(N), ];
                    Constraints = [Constraints   A_vars{k}'+A_vars{k} <= -epsilon*eye(N,N)];
                    Constraints = [Constraints,  P_var > epsilon*eye(N), b_vars{k} == -A_vars{k}*attractor];
                case 2
                    P_g = varargin{2};
                    % Option 1
                    Constraints = [Constraints   A_vars{k}'*(P_g'+ P_g) + (P_g'+ P_g)*A_vars{k} <= -epsilon*eye(N,N)];
                    
                    % Option 2
%                     Constraints = [Constraints, A_vars{k}' + A_vars{k} <= -epsilon*eye(N) ];
%                     Constraints = [Constraints  (P_g'+ P_g)*A_vars{k} <= -epsilon*eye(N,N)];
                    
                    % bias constraint
                    Constraints = [Constraints,  b_vars{k} == -A_vars{k}*attractor];
            end
    end    
    
    % Posterior Probabilities per local model
    h_k = posterior_probs_gmm(Xi_ref,gmm,'norm');    
    
    % Calculate our estimated velocities caused by each local behavior
    Xi_d_dot_c_raw = sdpvar(N,M,K, 'full');%zeros(size(Qd));
    for k = 1:K
            h_K = repmat(h_k(k,:),[N 1]);
            f_k = A_vars{k}*Xi_ref + repmat(b_vars{k},[1 M]);
%             f_k = A_vars{k}*Xi_ref;
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
        % Assigning initial guess for variables from convex problem
        for k=1:K
            assign(A_vars{k},A0(:,:,k));
        end
    end
    
    % Solve optimization problem
    sol = optimize(Constraints, Objective, sdp_options)
    if sol.problem ~= 0
        yalmiperror(sol.problem);
    end
    
    for k = 1:K
        A_c(:,:,k) = value(A_vars{k});  
        b_c(:,k)   = value(b_vars{k});  
    end    
    
end

if exist('P_var','var')
    P = value(P_var);
else
    P = eye(N);
end
check(Constraints)
fprintf('Total error: %2.2f\nComputation Time: %2.2f\n', value(Objective),sol.solvertime);

end