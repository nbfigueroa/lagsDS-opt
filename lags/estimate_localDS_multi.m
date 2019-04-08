function [A_l, b_l, A_d, b_d] = estimate_localDS_multi(Data, A_g, att_g, fl_type, att_l, Q, Mu, Sigma, Norm, limits, stability_vars)
P_g            = stability_vars.P_g;
P_l            = stability_vars.P_l;
alpha_fun      = stability_vars.alpha_fun;
h_fun          = stability_vars.h_fun;
grad_h_fun     = stability_vars.grad_h_fun;
activ_fun      = stability_vars.activ_fun;
grad_alpha_fun = stability_vars.grad_alpha_fun;
grad_lyap_fun  = stability_vars.grad_lyap_fun ;
lambda_fun     = stability_vars.lambda_fun ;

state_dim      = size(Data,1)/2;

if stability_vars.add_constr
    % Draw Initial set of samples for point-wise stability constraints
    desired_samples       = stability_vars.init_samples;
    desired_alpha_contour = 0.99;
    desired_Gauss_contour = -Norm*(desired_alpha_contour-1);
    chi_samples_out       = draw_chi_samples (Sigma,Mu,round(desired_samples/2), stability_vars.activ_fun, 'isocontours', desired_Gauss_contour);
    desired_Gauss_contour = -(Norm*0.005)*(desired_alpha_contour-1);
    chi_samples_in        = draw_chi_samples (Sigma,Mu,round(desired_samples/2), stability_vars.activ_fun, 'isocontours', desired_Gauss_contour);
    chi_samples           = [chi_samples_in chi_samples_out Data(1:state_dim,:)];
    
    % Local DS Optimization with Constraint Rejection Sampling
    stability_ensured = 0; iter = 1; max_iter = 5;
    while ~stability_ensured
        if iter == 1
            fprintf('First iteration, using %d BOUNDARY chi-samples..\n', size(chi_samples,2));
        else
            fprintf('iter = %d, using %d chi-samples..\n', iter, size(chi_samples,2));
        end
        pause(1);
        
        % Feed chi_samples to optimization structure
        stability_vars.chi_samples = chi_samples;
        
        % Full condition - less conservative estimation
        [A_l, b_l, A_d, b_d] = optimize_localDS_for_LAGS_multi(Data, A_g, att_g, att_l, Q, stability_vars);

        
        % Create function handles for current estimate of f_Q, grad_fQ
        clear f_Q
        V_Qk = @(x)fQk_constraint_multi(x, att_g, att_l, alpha_fun, h_fun, A_g, A_l, A_d, grad_lyap_fun, grad_h_fun,lambda_fun);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%  Check for violations in compact set by grid-sampling  %%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Plot Current Lyapunov Constraints function fQ
        if stability_vars.do_plots
            contour = 0;
            if exist('h_lyap','var'); delete(h_lyap);  end
            h_lyap = plot_lyap_fct(V_Qk, contour, limits,'Test $V_Q^k(\xi)$ with Grid-Sampling',1);        hold on;
            if exist('h_samples_used','var'); delete(h_samples_used);  end
            if contour
                h_samples_used = scatter(chi_samples(1,:),chi_samples(2,:),'+','c');
            else
                h_samples_used = scatter3(chi_samples(1,:),chi_samples(2,:),V_Qk(chi_samples),'+','c');
            end
        end
        cv_options                   = [];
        cv_options.chi_params        = struct('Mu',Mu,'Sigma',Sigma);
        cv_options.num_samples       = 10^5;
        cv_options.epsilon           = stability_vars.epsilon;
        tic;
        [constraint_violations_grid, max_fQ_grid, max_X_grid] = search_constraintViolations(V_Qk, activ_fun, cv_options);
        toc;
        
        % If we still have violations sample new contraints-eval points
        new_samples = []; num_grid_violations = length(constraint_violations_grid);
        if max_fQ_grid > -stability_vars.epsilon
            fprintf(2, 'Maxima in compact set is positive (f_max=%2.8f)! Current form is Not Stable!\n', max_fQ_grid);
            if num_grid_violations <= stability_vars.iter_samples
                new_samples = constraint_violations_grid;
            else
                num_max_samples  = round(stability_vars.iter_samples/2);
                num_rand_samples = stability_vars.iter_samples - num_max_samples;               
                new_samples = [new_samples constraint_violations_grid(:,1:num_max_samples)];
                constraint_violations_grid(:,1:num_max_samples) = [];
                new_samples = [new_samples constraint_violations_grid(:,randsample(length(constraint_violations_grid),num_rand_samples))];
            end
            if stability_vars.do_plots
                if exist('h_new_samples','var'); delete(h_new_samples);  end
                if contour
                    h_new_samples = scatter(new_samples(1,:),new_samples(2,:),'+','r');
                else
                    h_new_samples = scatter3(new_samples(1,:),new_samples(2,:),V_Qk(new_samples),'+','r');
                end
            end
            chi_samples = [chi_samples new_samples];
        else
            
            fprintf('Maxima in compact set is negative (f_max=%2.8f)! Stability is ensured!\n', max_fQ_grid);
            stability_ensured = 1;
        end
        
        % Constraint Sampling Loop
        iter = iter + 1;
        
        if iter > max_iter
            fprintf(2,'********** Locally Active DS is not feasible in this region! **************\n');
            stability_ensured = 1;
        end
            
    end
    
else
    tic;
    [A_l, b_l, A_d, b_d] = optimize_localDS_for_LAGS(Data, A_g, att_g, fl_type, stability_vars);
    toc;
end




end