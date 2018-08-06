function [chi_samples] = draw_chi_samples (Sigma,Mu,desired_samples,activ_fun)

num_samples = 0;
chi_samples = [];
Sigma
[V, L] = eig(Sigma);
while num_samples < desired_samples
    chi_samples_ = draw_from_ellipsoid( V * 4*L * V', Mu, desired_samples )';    
    
    % Upper Cut
    chi_samples_alpha = activ_fun(chi_samples_);
    id_ = chi_samples_alpha < 0.999;
    chi_samples_ = chi_samples_(:,id_);
    
    % Lower Cut
    chi_samples_alpha = activ_fun(chi_samples_);
    id_ = chi_samples_alpha > 0.35;
    chi_samples_ = chi_samples_(:,id_);                        
    chi_samples = [chi_samples chi_samples_];
    
    % Check that points are not overlapping
    chi_samples = unique(chi_samples.','rows').';    
    num_samples = length(chi_samples);    
end


end


