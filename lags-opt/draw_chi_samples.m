function [chi_samples] = draw_chi_samples (Sigma,Mu,desired_samples,activ_fun, varargin)

sampling_type = 'ellipsoid';
if nargin == 6
    sampling_type = varargin{1};
    c = varargin{2};
end

num_samples = 0;
chi_samples = [];
[V, L] = eig(Sigma);
switch sampling_type
    case 'ellipsoid'
        while num_samples < desired_samples
            % This function draws points uniformly from an n-dimensional ellipsoid
            % with edges and orientation defined by the the covariance matrix covmat.            
            chi_samples_ = draw_from_ellipsoid( V * 4*L * V', Mu, desired_samples )';
            
            % Upper Cut
            chi_samples_alpha = activ_fun(chi_samples_);
            id_ = chi_samples_alpha < 0.99;
            chi_samples_ = chi_samples_(:,id_);
            
            % Lower Cut
            chi_samples_alpha = activ_fun(chi_samples_);
            id_ = chi_samples_alpha > 0.001;
            chi_samples_ = chi_samples_(:,id_);            
            chi_samples = [chi_samples chi_samples_];
            
            % Check that points are not overlapping           
            if (desired_samples) < 100 && (desired_samples > 1)
                idx_close = find(squareform(pdist(chi_samples')) + eye(length(chi_samples)) < 0.05);
                length(idx_close)
                if ~isempty(idx_close)
                    [idx_x,idx_y] = ind2sub([length(chi_samples) length(chi_samples)],idx_close);
                    chi_samples(:,idx_x) = [];
                end
            end
            num_samples = length(chi_samples);
        end
        if num_samples > 5
            chi_samples = chi_samples(:,randsample(num_samples,desired_samples));
        end
    case 'isocontours'
            chi_samples = draw_from_isocontour( Sigma, Mu, desired_samples, c );     
end
end



