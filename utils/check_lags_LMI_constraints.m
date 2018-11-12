function [stable_necc, stab_local_contr, Big_Q_sym] = check_lags_LMI_constraints(x_test, alpha_fun, h_fun, A_g, A_l, A_d, att_g, att_l, P_l, P_g, lyap_der, Mu, Sigma)
% Activation functions
lyap_der_term = lyap_der(x_test);

alpha      = alpha_fun(x_test);
lyap_local = (x_test - att_g)'*P_l*(x_test - att_l);
if lyap_local >= 0 
    beta = 1; 
else
    beta = 0;
end
beta_l_2 = beta*2*lyap_local;
if h_fun(x_test) >= 1
    h_mod = 1;
else
    h_mod = h_fun(x_test);
end
A_L = h_mod*A_l + (1-h_mod)*A_d;

% Computing Auxiliary Matrices
Q_g  = A_g'*P_g + P_g*A_g;
Q_gl = A_g'*P_l;
Q_lg = A_L'*(2*P_g);
Q_l  = A_L'*P_l;

% Computing Block Matrices
Q_G = alpha * ( Q_g + beta_l_2*Q_gl );
Q_LG = (1-alpha)*( Q_lg + beta_l_2*Q_l );
Q_GL = alpha*beta_l_2*Q_gl;
Q_L  = (1-alpha)*beta_l_2*Q_l;
Q_LGL = Q_LG + Q_GL;

% Analysis with Block Matrix using Augmented States
xi_aug  = [x_test - att_g; x_test - att_l];

% Total Block Matrix
Big_Q    = [Q_G zeros(2,2); Q_LGL Q_L];

% Symmetric form of Big Q (Compute analytically)       
Big_Q_sym = [Q_G 0.5*Q_LGL'; 0.5*Q_LGL 0.5*(Q_L+Q_L')];
lambda_Q  = eig(Big_Q_sym);

% Symmetric form of Big Q  with Schur Complement Analysis         
A_Q = Q_G; B_Q = 0.5*Q_LGL'; C_Q = 0.5*(Q_L+Q_L');
S_Q = C_Q - B_Q'*inv(A_Q)*B_Q;
lambda_A      = eig(A_Q);
lambda_S      = eig(S_Q);
lambda_C      = eig(C_Q);
lambda_B      = eig(B_Q);
lambda_BAB    = eig(B_Q'*inv(A_Q)*B_Q);
F_Q = [eye(2) inv(A_Q)*B_Q; zeros(2,2) eye(2)]; 
G_Q = [A_Q zeros(2,2); zeros(2,2) S_Q];

% Necessary Stability Condition
quad_term          = xi_aug'*Big_Q*xi_aug;
quad_term_sym      = xi_aug'*Big_Q_sym*xi_aug;


fprintf('Big-Q Quadratic term: %3.3f Symmetric: %3.3f and Lyapunov Condition: %3.3f\n', quad_term, quad_term_sym, lyap_der_term);

% Decomposing Indefinite Matrix into sum of PD Method 1
lambda_Q_sym_dec  = eig(G_Q);
tI = abs(min(lambda_Q_sym_dec))*eye(4); 
P_pos =  G_Q + 1.5*tI; P_neg = 1.5*tI;
xi_aug_F = F_Q*xi_aug;
quad_term_sym_dec_ind =  xi_aug_F'*P_pos*xi_aug_F - xi_aug_F'*P_neg*xi_aug_F;
fprintf('Decomposed Quadratic Term: %3.3f ==> P+: %3.3f and P-: %3.3f\n', quad_term_sym_dec_ind, xi_aug_F'*P_pos*xi_aug_F, xi_aug_F'*P_neg*xi_aug_F);

% Inside the compact set \Chi
D_M  = sqrt((x_test-Mu)'*Sigma^(-1)*(x_test-Mu));
inside_set = D_M < 2;
fprintf('Inside Compact set: %d \n',inside_set);

% Case 1: Reduces to global DS
if (beta == 1 && alpha > 0.98)
    fprintf('CASE 1: Special Symmetric (B=BT) Saddle Point Matrix with alpha=%2.3f + beta=%2.3f\n',alpha,beta)               
   % Using saddle-point theorem
    J_Q = [eye(2,2) zeros(2,2); zeros(2,2) -eye(2,2)];
    Big_Q_sym_hat   = J_Q*Big_Q_sym;
    lyap_term_hat = xi_aug'*Big_Q_sym_hat*xi_aug  + [x_test - att_l]'*2*B_Q'*[x_test - att_g]       
    g_term =    [x_test - att_g]'*Q_G*[x_test - att_g];
    gl_term =   [x_test - att_l]'*Q_LGL*[x_test - att_g];     
    fprintf('Lyapunov Term decomposition g=%2.4f + lg=%2.4f = %2.4f \n',g_term, gl_term, g_term + gl_term);        
    sign_BAB = 0;
end

% Case 2 for local DS
if (beta == 0 && alpha < 0.98)
    fprintf('CASE 2: Special Non-Symmetric (B~=BT) Saddle Point Matrix with alpha=%2.3f + beta=%2.3f\n',alpha,beta)                   
    % Using saddle-point theorem
    J_Q = [eye(2,2) zeros(2,2); zeros(2,2) -eye(2,2)];
    Big_Q_sym_hat   = J_Q*Big_Q_sym;          
    neg_term_hat = xi_aug'*Big_Q_sym_hat*xi_aug;        
    ind_term_hat = [x_test - att_l]'*2*B_Q'*[x_test - att_g];    
    lyap_term_hat = neg_term_hat  + ind_term_hat;
    fprintf('Lyapunov Term decomposition g=%2.4f + lg=%2.4f == %2.4f \n',neg_term_hat, ind_term_hat, lyap_term_hat);
    
    sign_Q_hat = checkDefiniteness(Big_Q_sym_hat);
    stab_local_contr = neg_term_hat  < -ind_term_hat;
    fprintf(2, 'Proposed Stability condition CASE 2: %d \n', stab_local_contr);
    sign_BAB = 0;
end

% Case 3 for local DS
if (beta > 0 && alpha < 0.99)
    fprintf('CASE 3: Generalized Saddle Point Matrix with alpha=%2.3f + beta=%2.3f\n',alpha,beta)
    J_Q = [eye(2,2) zeros(2,2); zeros(2,2) -eye(2,2)];
    Q_BAB_hat = J_Q*[A_Q B_Q; B_Q' zeros(2,2)];
    neg_term_hat = xi_aug'*Q_BAB_hat*xi_aug;
    ind_term_hat   =  [x_test - att_l]'*2*B_Q'*[x_test - att_g] + [x_test - att_l]'*C_Q*[x_test - att_l];
    lyap_term_dec_hat = neg_term_hat + ind_term_hat ;
    fprintf('Lyapunov Term decomposition neg=%2.4f + ind=%2.4f == %2.4f \n',neg_term_hat, ind_term_hat, lyap_term_dec_hat);
    
    sign_BAB = checkDefiniteness(Q_BAB_hat);
    stab_local_contr = neg_term_hat  < -ind_term_hat;
    fprintf(2, 'Proposed Stability condition CASE 3: %d \n', stab_local_contr);
end

stable_necc = [x_test - att_g]'*Q_G*[x_test - att_g] < -[x_test - att_l]'*Q_LGL*[x_test - att_g] + [x_test - att_l]'*Q_L*[x_test - att_l];
fprintf(2, 'Proposed Stability condition: %d \n', stable_necc);

end