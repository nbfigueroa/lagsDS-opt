function [Q] = quadratic_4d_schur_optimal(x, att_l,  S)

[N,M] = size(x);
Q = zeros(1,M);
for i=1:M
    Y = x(:,i) - att_l;
    Q(i) = Y'*S*Y;
end