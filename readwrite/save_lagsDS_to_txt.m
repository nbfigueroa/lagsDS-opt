function save_lagsDS_to_txt(DS_name, pkg_dir, ds_gmm, A_g,  att_g, A_l_k, A_d_k, att_l, w_l, b_l, scale, b_g)

model_dir = strcat(pkg_dir,'/models/',DS_name, '/');
mkdir(model_dir)

% GMM parameters
Priors = ds_gmm.Priors;
Mu     = ds_gmm.Mu;
Sigma  = ds_gmm.Sigma;

% Writing Dimensions
Dimensions = [length(Priors); size(Mu,1)];
dlmwrite(strcat(model_dir,'dimensions'), Dimensions,'Delimiter',' ','precision','%.8f');

%%%%%%%%%%%%%%%% Global DS Parameters %%%%%%%%%%%%%%%%
% Writing attractor
dlmwrite(strcat(model_dir,'att_g'), att_g,'Delimiter',' ','precision','%.8f');

% Writing Priors
dlmwrite(strcat(model_dir,'Priors'), Priors,'Delimiter',' ','precision','%.8f');

% Writing Mu
dlmwrite(strcat(model_dir,'Mu'), Mu, 'newline','unix','Delimiter',' ','precision','%.8f');

% Writing Sigma
for i=1:length(Priors)
    dlmwrite(strcat(model_dir,'Sigma'), Sigma(:,:,i),'newline','unix','-append','Delimiter',' ','precision','%.8f');    
end

% Writing Ag's
for i=1:length(Priors)   
    dlmwrite(strcat(model_dir,'A_g'), A_g(:,:,i),'newline','unix','-append','Delimiter',' ','precision','%.8f');
end

%%%%%%%%%%%%%%%% Local DS Parameters %%%%%%%%%%%%%%%%
% Writing Al's
for i=1:length(Priors)   
    dlmwrite(strcat(model_dir,'A_l'), A_l_k(:,:,i),'newline','unix','-append','Delimiter',' ','precision','%.8f');
end

% Writing Ad's
for i=1:length(Priors)   
    dlmwrite(strcat(model_dir,'A_d'), A_d_k(:,:,i),'newline','unix','-append','Delimiter',' ','precision','%.8f');
end

% Writing att_l
dlmwrite(strcat(model_dir,'att_l'), att_l, 'newline','unix','Delimiter',' ','precision','%.8f');

% Writing w_l (Hyper-plane function basis)
dlmwrite(strcat(model_dir,'w_l'), w_l, 'newline','unix','Delimiter',' ','precision','%.8f');

% Writing b_l (lambda function parameter)
dlmwrite(strcat(model_dir,'b_l'), b_l,'Delimiter',' ','precision','%.8f');

% Writing scale (scale parameter)
dlmwrite(strcat(model_dir,'scale'), scale,'Delimiter',' ','precision','%.8f');

% Writing b_g (breadth of global radius function)
dlmwrite(strcat(model_dir,'b_g'), b_g,'Delimiter',' ','precision','%.8f');

