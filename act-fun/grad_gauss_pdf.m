function [grad_gauss] = grad_gauss_pdf(x, Mu, Sigma)

[N,M] = size(x);

% Compute probabilities p(x^i|Mu,Sigma)
Px_theta = ml_gaussPDF(x, Mu, Sigma) + eps;
grad_gauss = zeros(N,M);
for i=1:M
    grad_gauss(:,i) = Px_theta(i) * inv(Sigma) * (x(:,i)- Mu);
end



end