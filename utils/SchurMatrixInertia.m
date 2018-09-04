function [Inertia_H] = SchurMatrixInertia(H, M)
% Computes the matrix Intertia of a partitioned block Hermitain matrix
% Using the Haynsworth Inertia Additivity Formula
% * Haynsworth, E. V., "Determination of the inertia of a partitioned Hermitian matrix", 
% Linear Algebra and its Applications, volume 1 (1968), pages 73–81
% https://en.m.wikipedia.org/wiki/Haynsworth_inertia_additivity_formula
%
% Hermitain matrix representation
% H = [H_11   H12
%      H_12*  H22]
% H \in R^MxM, H_{ii} \in R^MxM

[N, ~] = size(H);
H_11 = H(1:M,1:M);
H_12 = H(1:M,M+1:N);
H_22 = H(M+1:N,M+1:N);
H11_Schur = H_22 - H_12'*inv(H_11)*H_12;

In_H11        = computeMatrixInertia(H_11);
In_H11_Schur  = computeMatrixInertia(H11_Schur);

Inertia_H = In_H11 + In_H11_Schur;

end

function [Inertia] = computeMatrixInertia(A, N)
lambda_A = eig(A);
Inertia = zeros(1,3);
Inertia(1,1) = sum(lambda_A > 0);
Inertia(1,2) = sum(lambda_A < 0);
Inertia(1,3) = sum(lambda_A == 0);
end
