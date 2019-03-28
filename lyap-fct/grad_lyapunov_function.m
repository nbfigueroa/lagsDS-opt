function [grad_lyap] = grad_lyapunov_function(x,attractor,P)
% Auxiliary Variables
[N,M]    = size(x);
% Output Variables           
grad_lyap = zeros(M,N);
if (sum(sum(P==eye(2)))/(N*2)) ==1 
    for i = 1:M
        grad_lyap(i,:) = (x(:,i)-attractor); 
    end        
else      
    for i = 1:M
        grad_lyap(i,:) = (x(:,i)-attractor)'*P; 
    end             
end
grad_lyap = grad_lyap';
end