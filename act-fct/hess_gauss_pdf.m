function [hess_gauss] = hess_gauss_pdf(x, Mu, Sigma)

[N,M] = size(x);

% Compute probabilities p(x^i|Mu,Sigma)
Px_theta = ml_gaussPDF(x, Mu, Sigma) + eps;
hess_gauss = zeros(N,N,M);
for i=1:M           
    hess_gauss(:,:,i) = Px_theta(i)* ((inv(Sigma)*(x(:,i)- Mu))*((x(:,i)- Mu)'*inv(Sigma))  - inv(Sigma));
end

end