function [A_l, b_l, A_d, b_d] = estimate_localDS(Data, A_g, att_g, fl_type, att_l, Mu, Sigma, Norm, limits, stability_vars)

test_grid      = stability_vars.test_grid;
P_g            = stability_vars.P_g;
P_l            = stability_vars.P_l;
alpha_fun      = stability_vars.alpha_fun;
h_fun          = stability_vars.h_fun;
grad_h_fun     = stability_vars.grad_h_fun;
activ_fun      = stability_vars.activ_fun;
grad_alpha_fun = stability_vars.grad_alpha_fun;

if stability_vars.add_constr
    % Draw Initial set of samples for point-wise stability constraints
    desired_samples       = stability_vars.init_samples;
    desired_alpha_contour = 0.99;
    desired_Gauss_contour = -Norm*(desired_alpha_contour-1);
    chi_samples = draw_chi_samples (Sigma,Mu,desired_samples, stability_vars.activ_fun, 'isocontours', desired_Gauss_contour);
    
    % Local DS Optimization with Constraint Rejection Sampling
    stability_ensured = 0; iter = 1;
    while ~stability_ensured
        if iter == 1
            fprintf('First iteration, using %d BOUNDARY chi-samples..\n', size(chi_samples,2));
        else
            fprintf('iter = %d, using %d chi-samples..\n', iter, size(chi_samples,2));
        end
        pause(1);
        
        % Feed chi_samples to optimization structure
        stability_vars.chi_samples = chi_samples;
        % Run optimization with chi-sample constraints
        if strcmp(stability_vars.constraint_type,'hessian')
            % Approximate - very conservative estimation
            [A_l, b_l, A_d, b_d, gamma] = optimize_localDS_for_LAGS_Hess(Data, A_g, att_g, stability_vars);
        else
            % Full condition - less conservative estimation
            [A_l, b_l, A_d, b_d] = optimize_localDS_for_LAGS(Data, A_g, att_g, fl_type, stability_vars);
        end
        
        % Create function handles for current estimate of f_Q, grad_fQ
        clear f_Q
        f_Q = @(x)fQ_constraint_single(x, att_g, att_l, P_g, P_l, alpha_fun, h_fun, grad_h_fun, A_g, A_l, A_d);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%  Check for violations in compact set by grid-sampling  %%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Plot Current Lyapunov Constraints function fQ
        if stability_vars.do_plots
            contour = 0;
            if exist('h_lyap','var'); delete(h_lyap);  end
            h_lyap = plot_lyap_fct(f_Q, contour, limits,'Test $f_Q(\xi)$ with Grid-Sampling',1);        hold on;
            if exist('h_samples_used','var'); delete(h_samples_used);  end
            if contour
                h_samples_used = scatter(chi_samples(1,:),chi_samples(2,:),'+','c');
            else
                h_samples_used = scatter3(chi_samples(1,:),chi_samples(2,:),f_Q(chi_samples),'+','c');
            end
        end
        cv_options                   = [];
        cv_options.chi_params        = struct('Mu',Mu,'Sigma',Sigma);
        cv_options.num_samples       = 10^5;
        cv_options.epsilon           = stability_vars.epsilon;
        tic;
        [constraint_violations_grid, max_fQ_grid, max_X_grid] = search_constraintViolations(f_Q, activ_fun, cv_options);
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
                    h_new_samples = scatter3(new_samples(1,:),new_samples(2,:),f_Q(new_samples),'+','r');
                end
            end
            chi_samples = [chi_samples new_samples];
        else
            
            fprintf('Maxima in compact set is negative (f_max=%2.8f)! Stability is ensured!\n', max_fQ_grid);
            
            % Check via optimization if parameters are stable
            if test_grid
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%% DOUBLE CHECK! Find maxima in compact set with Gradient Ascent on estimated fQ() %%%%
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Gradient of Lyapunov Function
                clear grad_fQ
                grad_fQ = @(x)gradient_fQ_constraint_single(x, att_g, att_l, P_g, P_l, alpha_fun, grad_alpha_fun, h_fun, grad_h_fun, A_g, A_l, A_d);
                % Plot Current Lyapunov Constraints function fQ
                if stability_vars.do_plots
                    plot_lyap_fct(f_Q, 1, limits, 'Estimated $f_Q(\xi)$',1); hold on;
                    plot_gradient_fct(grad_fQ, limits,  'Estimated $f_Q$ with $\nabla_{\xi}f_Q$');
                end
                % Do the maxima search with gradient ascent
                lm_options                = [];
                lm_options.type           = 'grad_ascent';
                lm_options.num_ga_trials  = 5;
                lm_options.do_plots       = stability_vars.do_plots;
                lm_options.init_set       = chi_samples;
                lm_options.verbosity      = 0;
                [local_max, local_fmax]   = find_localMaxima(f_Q, grad_fQ, lm_options);
                [max_val, max_id]         = max(local_fmax);
                
                % Compare f_Q - mod_term < 0                
                if max_val > 0.05
                    fprintf (2, 'There was an fQ_max(%2.8f) > 0 at x=%3.3f,y=%3.3f found :( !!\n', max_val, local_max(1,max_id),local_max(2,max_id));
                    fprintf (2, 'You must re-run optimization, try increasing epsilon! \n');
                else
                    fprintf ('+++++ ALL fQ_max < 0 +++++!!\n');
                    % Estimated parameters ensure stability as fQ_max < 0
                    stability_ensured = 1;
                end
                fprintf('Optimization converged to a stable solution with %d chi_samples in %d iterations!\n', size(chi_samples,2),iter);
            end
        end
        
        % Constraint Sampling Loop
        iter = iter + 1;
    end
    
else
    tic;
    [A_l, b_l, A_d, b_d] = optimize_localDS_for_LAGS(Data, A_g, att_g, fl_type, stability_vars);
    toc;
end




end