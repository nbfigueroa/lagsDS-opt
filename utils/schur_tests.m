%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Testing Schur Complement and Definiteness %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% For Postive Definite Block Matrices
clc;
X_pos = hilb(4);
A_pos = X_pos(1:2,1:2);
B_pos = X_pos(1:2,3:4);
C_pos = X_pos(3:4,3:4);
S_pos = C_pos - B_pos'*pinv(A_pos)*B_pos;

lambda_X = eig(X_pos)
lambda_A = eig(A_pos)
lambda_S = eig(S_pos)

%% For Negative Definite Block Matrices
clc;
X_pos = -hilb(4);
A_pos = X_pos(1:2,1:2);
B_pos = X_pos(1:2,3:4);
C_pos = X_pos(3:4,3:4);
S_pos = C_pos - B_pos'*pinv(A_pos)*B_pos;

lambda_X = eig(X_pos)
lambda_A = eig(A_pos)
lambda_S = eig(S_pos)