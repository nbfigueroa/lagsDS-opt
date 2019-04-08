function [Mu] =  kmeans_init(X, k, init)
%KMEANS_INIT This function computes the initial values of the centroids
%   for k-means algorithm, depending on the chosen method.
%
%   input -----------------------------------------------------------------
%   
%       o X     : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o k     : (double), chosen k clusters
%       o init  : (string), type of initialization {'random','uniform'}
%
%   output ----------------------------------------------------------------
%
%       o Mu    : (N x k), an Nxk matrix where the k-th column corresponds
%                          to the k-th centroid mu_k \in R^N                   
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Auxiliary Variable
[N, M] = size(X);

% Output Variable
Mu = zeros(N, k);

switch init
    case 'random'  % Select K datapoints at random from X        
        Mu = X(:,randsample(M,k));
        
        % or
        
        % Mu = datasample(X,k,2);
        
    case 'uniform' % Select k datapoints uniformly at random from the range of X                

        % Xmins = min(X,[],2);
        % Xmaxs = max(X,[],2);
        % Mu = Xmin_values(:,ones(1,k)) + rand(k,N)'.*(Xmaxs(:,ones(1,k))-Xmins(:,ones(1,k)))
        
        % or
        
        Xmin_values = min(X,[],2);
        Xrange_values = range(X,2);
        Mu = Xmin_values(:,ones(1,k)) + rand(k,N)'.*(Xrange_values(:, ones(1,k)));
        
    case 'plus' % k-means++ algorithm
        
        % Select the first centroid by sampling uniformly at random (like 'random')        
        Mu(:,1) = X(:,randsample(M,1));
        minDistances = ones(1,M)*10000;

       % Select the rest of the Centroids by a probabilistic model
       % clc;
       for ii = 2:k                    
       
            minDistances = min(minDistances,my_distX2Mu(X,Mu(:,ii-1), 'L2'));           
            denominator = sum(minDistances);
            Probs = minDistances/denominator            
            Mu(:,ii) = datasample(X,1,2,'Weights',Probs);                    
       end    
       
    otherwise
        warning('Unexpected initialization type. No centroids computed.')
end

end