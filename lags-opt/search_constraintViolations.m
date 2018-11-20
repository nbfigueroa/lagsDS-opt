function [constraint_violating_samples, max_fQ, max_x] =  search_constraintViolations(f_Q, activ_fun, cv_options)

% Parse search option selected
type = cv_options.type;

switch type
    case 'grid'
        % Parse parameters for search option
        num_samples = cv_options.num_samples;
        chi_params  = cv_options.chi_params;
        Mu          = chi_params.Mu;
        Sigma       = chi_params.Sigma;
        
        fprintf(2, 'Checking violation of Lyapunov Constraint on %d Samples.. ', num_samples);        
        % Randomly Sample Points to evaluate
        chi_samples = draw_chi_samples (Sigma, Mu, num_samples, activ_fun);
        fprintf(2, '. done. \n');
        
        % Evaluate Samples
        lyap_constr_samples = f_Q(chi_samples);
        
        % Necessary Constraints
        violations                   = lyap_constr_samples >= 0;
        constraint_violating_samples = [];
        constraint_violating_samples = chi_samples(:,violations);

        if any(violations)
            if length(violations)>1
                fQ_constraint_violations = f_Q(constraint_violating_samples);
                [fQ_sorted,fQ_id] = sort(fQ_constraint_violations,'descend');
                constraint_violating_samples = constraint_violating_samples(:,fQ_id);
                max_x  = constraint_violating_samples(:,fQ_id(1));
                max_fQ = fQ_sorted(1);
            else
                max_x  = constraint_violating_samples;
                max_fQ = f_Q(constraint_violating_samples);
            end
            fprintf(2, '%d/%d violations of Lyapunov Constraint (fQ_max=%2.6f)!!\n', length(fQ_constraint_violations),num_samples,max_fQ);
        else
            
            [max_fQ, max_fQ_id] = max(lyap_constr_samples);
            max_x = chi_samples(:,max_fQ_id);
            fprintf(2,'No violations out of %d samples!!!\n',num_samples);
        end
    case 'grad_ascent'
        % Parse parameters for search option
        num_ga_trials = cv_options.num_ga_trials;
        init_set      = cv_options.init_set;
        grad_fQ       = cv_options.grad_fQ;
        
        ga_options = [];
        ga_options.plot     = cv_options.do_plots;  % plot init/final and iterations        
        ga_options.gamma    = 0.0001;               % step size (learning rate)
        ga_options.max_iter = 2000;                 % maximum number of iterations
        ga_options.f_tol    = 1e-10;                % termination tolerance for F(x)        
        ga_options.verbose  = cv_options.do_plots;  % Show values on iterations
        opt_xmax = zeros(2,num_ga_trials);
        opt_fmax = zeros(1,num_ga_trials);
        
        for n=1:num_ga_trials
            x0 = init_set(:,randsample(length(init_set),1));
            fprintf('Finding maxima in Test function using Gradient Ascent init=%d...\n',n);
            [opt_fmax(n), opt_xmax(:,n), ~, ~, ~] = gradientAscent(f_Q,grad_fQ, x0, ga_options);
            if opt_fmax(n) > 0
                break;
            end
        end        
        max_violations = opt_fmax(n) >= 0;        
        
end


end
