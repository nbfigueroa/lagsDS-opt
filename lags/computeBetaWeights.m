function [beta_weights] = computeBetaWeights(Xi_ref, est_labels,att_g, att_l, P_l)
%% Find maximum beta values for each local DS
K = size(att_l,2);
for k=1:K
   x_local = Xi_ref(:,est_labels == k);
%    x_local = Xi_ref;
   for l=1:K
        % Estimate local assymetric lyapunov function values
        for i=1:length(x_local)
            lyap_local_k_ =   (x_local(:,i) - att_g)'*P_l(:,:,l)*(x_local(:,i) - att_l(:,l));
        end
        lyap_local_k_ = [lyap_local_k_   (att_l(:,k) - att_g)'*P_l(:,:,l)*(att_l(:,k) - att_l(:,l))];
        
        % Get the maximum value
        lyap_local_k = max(lyap_local_k_);

        % Computing activation term
        if lyap_local_k >= 0
            beta = 1;
        else
            beta = 0;
        end
        beta_weights(k,l) = 2 * beta * lyap_local_k;       
   end    
end  
end