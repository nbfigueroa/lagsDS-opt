function [local_max, local_fmax] =  find_localMaxima(f, grad_f, lm_options)

% Parse search option selected
type = lm_options.type;
num_ga_trials = lm_options.num_ga_trials;
init_set      = lm_options.init_set;
verbosity     = lm_options.verbosity;
        
switch type
    case 'grad_ascent'

        
        % Set gradient ascent parameters
        ga_options = [];
        ga_options.gamma    = 0.0001;                % step size (learning rate)
        ga_options.max_iter = 1000;                  % maximum number of iterations
        ga_options.f_tol    = 1e-8;                  % termination tolerance for F(x)
        ga_options.plot     = lm_options.do_plots;   % plot init/final and iterations        
        ga_options.verbose  = verbosity;             % Show values on iterations
                      
        local_max  = zeros(2,num_ga_trials);
        local_fmax = zeros(1,num_ga_trials);
        for n=1:num_ga_trials
            % Set Initial value
            x0 = init_set(:,randsample(length(init_set),1));
            fprintf('Finding maxima in Test function using Gradient Ascent...\n');
            [local_fmax(1,n), local_max(:,n), ~, ~, ~] = gradientAscent(f,grad_f, x0, ga_options);
        end        
end


end
