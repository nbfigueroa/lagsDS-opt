function [in] = bin_alpha_set(x,alpha_fun)
alpha  = feval(alpha_fun,x);
in = zeros(1, length(alpha));
for i=1:length(alpha)
    in(i) = alpha(i) > 0.99;
end
end