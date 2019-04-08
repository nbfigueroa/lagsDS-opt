function save_lagsDS_to_Yaml(DS_name, pkg_dir,  ds_gmm, A_k, att, x0_all, dt, A_l_k, A_d_k, att_l, w_l, b_l, scale, b_g, gpr_filename)

% GMM parameters
K          = length(ds_gmm.Priors);
dim        = size(ds_gmm.Mu,1);
Priors_vec = ds_gmm.Priors;
Mu_vec     = ds_gmm.Mu(1:end);
Sigma_vec  = ds_gmm.Sigma(1:end);

% DS parameters
Ag_vec      = A_k(1:end);
Al_vec      = A_l_k(1:end);
Ad_vec      = A_d_k(1:end);

% Initial points (to simulate)
x0_all_vec = x0_all(1:end);

% Create structure to dump in yaml file
lagsDS_model =[];
lagsDS_model.name         = DS_name;
lagsDS_model.K            = K;
lagsDS_model.M            = dim;
lagsDS_model.Priors       = Priors_vec;
lagsDS_model.Mu           = Mu_vec;
lagsDS_model.Sigma        = Sigma_vec;
lagsDS_model.A_g          = Ag_vec;
lagsDS_model.att_g        = att(1:end);
lagsDS_model.A_l          = Al_vec;
lagsDS_model.A_d          = Ad_vec;
lagsDS_model.att_l        = att_l(1:end);
lagsDS_model.w_l          = w_l(1:end);
lagsDS_model.b_l          = b_l(1:end);
lagsDS_model.scale        = scale;
lagsDS_model.b_g          = b_g;
lagsDS_model.gpr_path     = gpr_filename;
lagsDS_model.x0_all       = x0_all_vec;
lagsDS_model.dt           = dt;

% Visualize what will be dumped on yaml file
lpvDS_dump = YAML.dump(lagsDS_model);
yamlfile = strcat(pkg_dir,'/models/', lagsDS_model.name,'.yml');

% Save yaml file
fprintf('The following parameters were saved in: %s \n', yamlfile);
disp(lpvDS_dump);
YAML.write(yamlfile,lagsDS_model)

end