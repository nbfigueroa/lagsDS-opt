function [h] = hyper_plane(x,w,intercept)

% Auxiliary Variables
[N,M]    = size(x);

% Bias
b = 1-w'*intercept;

% Output variable
h = zeros(1,M);
for i = 1:M        
    h_raw  = w'*x(:,i) + b ;       
    
    % Discrete way
%     h(:,i) = max(0,h_raw);
    
    % Continuous way
    h(:,i) = 1/2 * (h_raw + abs(h_raw));
    
end
end
