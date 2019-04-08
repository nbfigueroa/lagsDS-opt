function [ Priors0, Mu0, labels0, Sigma0 ] = my_gmmInit(X, K, cov_type)
%MY_GMMINIT Computes initial estimates of the parameters of a GMM 
% to be used for the EM algorithm
%   input------------------------------------------------------------------
%
%       o X         : (N x M), a data set with M samples each being of 
%                           dimension N, each column corresponds to a datapoint.
%       o K         : (1 x 1) number K of GMM components.
%       o cov_type  : string ,{'full', 'diag', 'iso'} type of Covariance matrix
%   output ----------------------------------------------------------------
%       o Priors0   : (1 x K), the set of priors (or mixing weights) for each
%                           k-th Gaussian component
%       o Mu0       : (N x K), an NxK matrix corresponding to the centroids 
%                           mu = {mu^1,...mu^K}
%       o labels0   : (1 x M), a vector of labels \in {1,..,k} 
%                           corresponding to the k-th Gaussian component
%       o Sigma0    : (N x N x K), an NxNxK matrix corresponding to the 
%                       Covariance matrices  Sigma = {Sigma^1,...,Sigma^K}
%%
% Auxiliary Variables
[N, M] = size(X);

%%%%%% STEP 1: Initialization of Priors, Means and Covariances %%%%%%
%%%%%% ===== IMPLEMENT STEP 1 HERE ===== %%%%%%
% Initialize Priors as Uniform Probabilities
Priors0 = ones(1,K)/sum(ones(1,K));

% Initialize Means with k-means
init='plus'; d_type='L2'; MaxIter = 100; plot_iter_k = 0;
[labels0, Mu0] =  my_kmeans(X, K, init, d_type, MaxIter, plot_iter_k);

% Initialize Sigmas with labels and Mu from k-means
Sigma0 = zeros(N,N,K);
for k=1:K 
        Sigma0(:,:,k) = my_covariance( X(:,labels0==k), Mu0(:, k), cov_type );
        
        % Add a tiny variance to avoid numerical instability
        Sigma0(:,:,k) = Sigma0(:,:,k) + 1E-5.*diag(ones(N,1));
end

end

