function [att_l, local_basis] = estimate_local_attractors_lags(Data, est_labels, ds_gmm, gauss_thres, radius_fun, att_g)


% Parse inputs
K  = length(ds_gmm.Priors);
Mu = ds_gmm.Mu;
Sigma = ds_gmm.Sigma;

% Auxiliary variables
[N,M] = size(Data);
N = N/2;
Data_k = [];
att_l     = zeros(N,K);
local_dirs = zeros(N,K);
local_basis = zeros(N,N,K);

for k=1:K
    % Extract local trajectories
    Data_k{k}      = Data(:,est_labels==k);
                      
    % Extract linear DS main directions with velocities    
    pos_data = Data_k{k}(1:2,:);
    vel_data = Data_k{k}(3:4,:);
    dist_global_att = find(radius_fun(Data_k{k}(1:2,:))' > 0.5);
    
    if ~isempty(dist_global_att)
        fprintf('%d-th Gaussian ends at the global attractor\n', k);
        local_dir = (att_g-Mu(:,k))/norm(att_g-Mu(:,k));
        local_dirs(:,k)  = local_dir;
        att_l(:,k)   = att_g;        
    else                
        % Compute weights for velocity vectors        
        vel_weights      = exp(-vecnorm(repmat(Mu(:,k),[1 length(pos_data)])-pos_data))/sum(exp(-vecnorm(repmat(Mu(:,k),[1 length(pos_data)])-pos_data))) ;      
        % Compute main velocity direction
        vel_dirs = vel_data./repmat((vecnorm(vel_data)+eps),[N 1]);   
        weighted_vel_dirs = repmat(vel_weights,[N 1]).*vel_dirs;
        local_dirs(:,k)  = sum(weighted_vel_dirs,2)./repmat(vecnorm(sum(weighted_vel_dirs,2)),[N 1]);
        
        % Perform "gradient" ascent to find local attractor
        v = local_dirs(:,k)/norm(local_dirs(:,k));
        gamma = 0.005; x = Mu(:,k);
        converged = 0; max_iter = 500; iter = 0;
        while iter < max_iter && ~converged
            x = x + gamma * v;
            prob = gaussPDF(x, Mu(:,k), Sigma(:,:,k));
            if prob < gauss_thres
                converged = 1;
            end
            iter = iter + 1;
        end
        att_l(:,k) = x;        
    end
    local_basis(:,:,k) = findDampingBasis(local_dirs(:,k));  
end


end