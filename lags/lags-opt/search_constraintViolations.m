function [constraint_violating_samples, max_fQ, max_x] =  search_constraintViolations(f_Q, activ_fun, cv_options)

% Parse parameters for search option
num_samples = cv_options.num_samples;
chi_params  = cv_options.chi_params;
Mu          = chi_params.Mu;
Sigma       = chi_params.Sigma;
epsilon     = cv_options.epsilon;

fprintf(2, 'Checking violation of Lyapunov Constraint on %d Samples.. ', num_samples);
% Randomly Sample Points to evaluate
chi_samples = draw_chi_samples (Sigma, Mu, num_samples, activ_fun);
fprintf(2, '. done. \n');

% Evaluate Samples
lyap_constr_samples = f_Q(chi_samples);

% Necessary Constraints
violations                   = lyap_constr_samples > -epsilon;
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

end
