function [Big_Q_sym] = construct_BigQ_sym(x_test, alpha_fun, h_fun, A_g, A_l, A_d, att_g, att_l, P_l, P_g)


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

% Symmetric form of Big Q (Compute analytically)       
Big_Q_sym = [Q_G 0.5*Q_LGL'; 0.5*Q_LGL 0.5*(Q_L+Q_L')];


end