function [signature] = checkDefiniteness(A)
% This (+,1,0) eigenvalues

M = size(A,1);
A_sym = 0.5*(A + A');
lambda_A = eig(A_sym);
signature = zeros(1,3);

if any(real(lambda_A) > 0)    
    signature(1,1) = sum(real(lambda_A) > 0);    
end

if any(real(lambda_A) < 0) 
    signature(1,2) = sum(real(lambda_A) < 0);
end

if any(real(lambda_A) == 0)
    signature(1,3) = sum(real(lambda_A) == 0);    
end

end