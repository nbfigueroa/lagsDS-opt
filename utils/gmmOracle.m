function [Priors, Mu, Sigma] = gmmOracle(X, true_labels)

labels = unique(true_labels);
K = length(labels);

% Auxiliary Variables
[N, M] = size(X);

% Initialize Priors, Mu, Sigmas with labels
Priors0 = zeros(1,K);
Mu0     = zeros(N,K);
Sigma0  = zeros(N,N,K);
for k=1:K 
    
        Priors0(1,k)  = sum(true_labels == labels(k))/M;        
        Mu0(:,k)      = mean(X(:,true_labels == labels(k)),2);                 
        Sigma0(:,:,k) = my_covariance( X(:,true_labels==labels(k)), Mu0(:, k), 'full' );
        
        % Add a tiny variance to avoid numerical instability
        Sigma0(:,:,k) = Sigma0(:,:,k) + 1E-5.*diag(ones(N,1));
end

[Priors, Mu, Sigma, ~] = my_gmmEM(X, K, 'full', Priors0, Mu0, Sigma0, 100);

end