function [x_dot] = linearDS(x, A, b)

if isempty(b)
    x_dot = A*x;
else
    x_dot = A*x + repmat(b,[1 length(x)]);
end

end