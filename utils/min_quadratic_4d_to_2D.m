function [x_min, Q_min, H_Q] = min_quadratic_4d_to_2D(att_g, att_l,  P)

A = P(1:2,1:2); B = P(1:2,3:4); C = P(3:4,3:4); B_T = P(3:4,1:2);
x_min =  att_l'*(B + B_T + 2*C)*inv(2*A + 2*(B+B_T) + 2*C);
x_min = x_min';
x_aug = [x_min - att_g; x_min - att_l];
Q_min = x_aug'*P*x_aug;

% Hessian of Current Quadratic function
H_Q = 2*A + 2*(B+B_T) + 2*C;