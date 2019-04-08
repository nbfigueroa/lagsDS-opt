function [A_l, b_l, A_d, b_d, num_grid_violations] = estimate_localDS_search(Data_k, kappa,A_g,  att_g, att_l, P_g, P_l, local_basis, alpha_fun, h_fun, grad_lyap,grad_h_fun, activ_fun, Mu, Sigma , lambda_fun, varargin )

            fprintf('\n==>Estimating local-DS with kappa=%2.3f\n', kappa);

            N = size(Data_k,1)/2;
%             w = -local_basis(:,1);
%             [A_l, ~, ~, ~] = estimate_localDS_known_gamma(Data_k, A_g,  att_g, att_l, 1, kappa , w, P_g, P_l, local_basis);
            R_gk = (local_basis(:,1)'*0.5*(A_g+A_g')*local_basis(:,1))/(local_basis(:,1)'*local_basis(:,1)); 
            
            % Scale for eigenvaues of local DS
            R_scale = 0.75;
            if nargin == 17
                DS_g = varargin{1};
                if DS_g
                    R_scale = 0.25;
                end
            end
            R_scale
            A_l = local_basis(:,:)*[R_scale*R_gk 0;0 R_scale*kappa*R_gk]*local_basis(:,:)'            
            b_l = -A_l*att_l;
            
            % Check Eigenvalues
            Lambda_l = eig(A_l);           
            
            % Construct the Deflective terms
            A_d = -0.5*min(Lambda_l)*eye(N);           
            b_d = -A_d*att_l;

            
            %%%%%%%%%%%%%% Verification on compact set %%%%%%%%%%%%%%
            % Create function handles for current estimate of f_Q, grad_fQ
            clear f_Q
            V_Qk = @(x)fQk_constraint_multi(x, att_g, att_l, alpha_fun, h_fun, A_g, A_l, A_d, grad_lyap, grad_h_fun, lambda_fun);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%  Check for violations in compact set by grid-sampling  %%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            cv_options                   = [];
            cv_options.chi_params        = struct('Mu',Mu,'Sigma',Sigma);
            cv_options.num_samples       = 10^5;
            cv_options.epsilon           = -10;
            tic;
            [constraint_violations_grid, max_fQ_grid, ~] = search_constraintViolations(V_Qk, activ_fun, cv_options);
            toc;
            
            % If we still have violations reduce kappa size
            num_grid_violations = length(constraint_violations_grid);
            if max_fQ_grid > -cv_options.epsilon
                fprintf(2, 'Maxima in compact set is positive (f_max=%2.8f)! Current form is Not Stable!\n', max_fQ_grid);
            else
                fprintf('Maxima in compact set is negative (f_max=%2.8f)! Stability is ensured!\n', max_fQ_grid);
            end   

end