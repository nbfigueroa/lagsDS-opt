function [] = simulate_passiveDS(hfig, robot, base, ds, target, dt, varargin)

% select our figure as gcf
% figure(hfig);
hold on

 
    % Setting robot to starting point
    disp('Select a starting point for the simulation...')
    disp('Once the simulation starts you can perturb the robot with the mouse to get an idea of its compliance.')
    
    infeasible_point = 1;
    while infeasible_point
        try
            xs = get_point(hfig) - base;

            qs = simple_robot_ikin(robot, xs);
            robot.animate(qs);
            infeasible_point = 0;
        catch
            warning('could not find a feasible joint space configuration. Please choose another point in the workspace.')
        end
    end
    
    % Run Simulation
    if nargin == 6
        [hd, hx] = simulation_passive_control(hfig, robot, base, ds, target, qs, dt);
    else
        struct_stiff = varargin{1};
        [hd, hx] = simulation_passive_control_cont(hfig, robot, base, ds, target, qs, dt, struct_stiff);
    end    
fprintf('Simulation ended.\n')

end

