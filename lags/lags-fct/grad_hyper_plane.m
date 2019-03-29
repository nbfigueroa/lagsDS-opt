function [grad_h] = grad_hyper_plane(x, w, h_fun)

% Auxiliary Variables
[N,M]    = size(x);

% Output variable
h = feval(h_fun,x);
grad_h = zeros(N,M);
for i = 1:M        
    if h(i) > 0.5
        grad_h(:,i) = w;
    end    
end
end
