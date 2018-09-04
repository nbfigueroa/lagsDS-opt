function [Q] = quadratic_4d_to_2D(x, att_g, att_l,  P)

[N,M] = size(x);
Q = zeros(1,M);
A = P(1:2,1:2); B = P(1:2,3:4); D = P(3:4,1:2); C = P(3:4,3:4);
for i=1:M
    X = x(:,i) - att_g;
    Y = x(:,i) - att_l;
    Q(i) = X'*A*X + Y'*D*X + X'*B*Y + Y'*C*Y;
end