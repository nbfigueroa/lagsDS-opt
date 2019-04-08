function [  Priors, Mu, Sigma, iter ] = my_gmmEM(X, K, cov_type, Priors0, Mu0, Sigma0, Max_iter)
%MY_GMMEM Computes maximum likelihood estimate of the parameters for the 
% given GMM using the EM algorithm and initial parameters
%   input------------------------------------------------------------------
%
%       o X         : (N x M), a data set with M samples each being of 
%                           dimension N, each column corresponds to a datapoint.
%       o K         : (1 x 1) number K of GMM components.
%       o cov_type  : string ,{'full', 'diag', 'iso'} type of Covariance matrix
%       o Priors0   : (1 x K), the set of INITIAL priors (or mixing weights) for each
%                           k-th Gaussian component
%       o Mu0       : (N x K), an NxK matrix corresponding to the INITIAL centroids 
%                           mu^(0) = {mu^1,...mu^K}
%       o Sigma0    : (N x N x K), an NxNxK matrix corresponding to the
%                    INITIAL Covariance matrices  Sigma^(0) = {Sigma^1,...,Sigma^K}
%       o Max_iter  : (1 x 1) maximum number of allowable iterations
%   output ----------------------------------------------------------------
%       o Priors    : (1 x K), the set of FINAL priors (or mixing weights) for each
%                           k-th Gaussian component
%       o Mu        : (N x K), an NxK matrix corresponding to the FINAL centroids 
%                           mu = {mu^1,...mu^K}
%       o Sigma     : (N x N x K), an NxNxK matrix corresponding to the
%                   FINAL Covariance matrices  Sigma = {Sigma^1,...,Sigma^K}
%       o iter      : (1 x 1) number of iterations it took to converge
%%

% Auxiliary Variables
[N, M] = size(X);
ll_old = -realmax;
eps = 1e-5;
% eps = realmin;
t_iter   = 0;
Px_k = zeros(K,M);
Pk_x = zeros(K,M);

% Stopping threshold for EM iterative update
ll_thres = 1e-6;

%%%%%% STEP 1: Initialization of Priors, Means and Covariances %%%%%%
Priors = Priors0;
Mu     = Mu0;
Sigma  = Sigma0;

while true
    
    %%%%%% STEP 2: Expectation Step: Membership probabilities %%%%%%
    
    % Compute probabilities p(x^i|k)
    for k=1:K
            Px_k(k,:) = ml_gaussPDF(X, Mu(:,k), Sigma(:,:,k));
    end
    
    %%% Compute posterior probabilities p(k|x) -- FAST WAY --- %%%
    alpha_Px_k = repmat(Priors',[1 M]).*Px_k;
    Pk_x = alpha_Px_k ./ repmat(sum(alpha_Px_k,1),[K 1]);    
    
    %%% Compute posterior probabilities p(k|x) -- SLOW WAY --- %%%
%     for i=1:M
%       Pk_x(:,i) = (Priors'.*Px_k(:,i))./(sum(Priors'.*Px_k(:,i)));
%     end      
    
    %%%%%% STEP 3: Maximization Step: Update Priors, Means and Sigmas %%%%%%    
    % Compute cumulated posterior probability
    Sum_Pk_x = sum(Pk_x,2);
    
    % Update Means and Covariance Matrix
    for k=1:K            
        
        % Update Priors
        Priors(k) = Sum_Pk_x(k)/M;
        
        % Update Means
        Mu(:,k) = X*Pk_x(k,:)' / Sum_Pk_x(k);         
                
        %%% Update Full Covariance Matrices  -- FAST WAY --- %%%
        % Demean Data
        X_ = bsxfun(@minus, X, Mu(:,k));                
        % Compute Full Sigma
        Sigma(:,:,k) = (repmat(Pk_x(k,:),N,1).*X_*X_')./ Sum_Pk_x(k); 
        
        %%% Update Full Covariance Matrices  -- SLOW WAY ---  %%%               
        % Sigma_ = zeros(N,N);
        % for i=1:M 
        %    Sigma_ = Sigma_ + (Pk_x(k,i) * (X(:,i)- Mu(:,k))*(X(:,i)- Mu(:,k))'); 
        % end
        % Sigma(:,:,k) = Sigma_/ Sum_Pk_x(k);        
        
        switch cov_type
            
            case 'full'
            
            case 'diag'
                Sigma(:,:,k) = diag(diag(Sigma(:,:,k)));
                
            case 'iso'
                sqr_dist = sum((X - repmat(Mu(:,k),1,M)).^2,1);                                
                Sigma(:,:,k) = eye(N,N)*(sqr_dist*Pk_x(k,:)' ./ Sum_Pk_x(k) ./N); 
            otherwise
                warning('Unexpected Covariance type. Using Full Covariance computed.')
                
        end        
        
        % Add a tiny variance to avoid numerical instability
        Sigma(:,:,k) = Sigma(:,:,k) + eps.*diag(ones(N,1));
    end    
    
    %%%%%% Stopping criterion %%%%%%
    ll = ml_LogLikelihood_gmm(X, Priors, Mu, Sigma);     
    if abs(ll_old - ll) < ll_thres
%             fprintf('Algorithm has converged at t_iter=%d with ll=%2.2f! Stopping EM.\n', t_iter, ll);
        break;
    end
    
    if t_iter >= Max_iter
%         fprintf('Maximum Iteration Reached t_iter=%d with ll=%2.2f! Stopping EM.\n', t_iter, ll);
        break;
    end
    
    ll_old = ll;
    t_iter = t_iter + 1;
end
iter= t_iter;
end

