function [grad_gmm] = grad_gmm_pdf(x, gmm)

% Unpack gmm
Mu     = gmm.Mu;
Priors = gmm.Priors;
Sigma  = gmm.Sigma;
K      = length(Priors);
[N,M] = size(x);

% Compute probabilities p(x^i|k)
for k=1:K
    Px_k(k,:) = ml_gaussPDF(x, Mu(:,k), Sigma(:,:,k)) + eps;
end

grad_gmm = zeros(N,M);
for i=1:M
    grad_x = zeros(N,1);
    for j=1:K
        grad_x = grad_x - Priors(j) * Px_k(j,i) * inv(Sigma(:,:,j)) * (x(:,i)- Mu(:,j));
    end
    grad_gmm(:,i) = grad_x;
end



end