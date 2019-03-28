function [] = simulate_passiveDS(hfig, robot, base, ds, target, dt, varargin)

% select our figure as gcf
% figure(hfig);
hold on

% set(hfig,'WindowButtonDownFcn',@(h,e)button_clicked(h,e));
% set(hfig,'WindowButtonUpFcn',[]);
% set(hfig,'WindowButtonMotionFcn',[]);
% hp = gobjects(0);

% Stop recording
% restart_btn = uicontrol('Position',[100 20 110 25],'String','restart',...
%               'Callback','uiresume(gcbf)');   
          
% while true
 
    % Setting robot to starting point
    disp('Select a starting point for the simulation...')
    disp('Once the simulation starts you can perturb the robot with the mouse to get an idea of its compliance.')
    
    infeasible_point = 1;
    while infeasible_point
        try
            xs = get_point(hfig) - base;
            % Another option (Start around demonstrations) :
            % xs  =  Data(1:2,1) - base + 0.15*randn(1,2)'
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

%     % Press Enter to start another simulation
%     clc;
%     fprintf('Press button to restart simulation.\n');
%     try
%         uiwait(gcf);
%         
%         % Remove Old Simulation if it exists
%         if exist('hd','var'), delete(hd); end
%         if exist('hx','var'), delete(hx); end
%         fprintf('Starting new simulation.\n')
%     catch
%         fprintf('figure closed.\n')
%         break;
%     end
% end
fprintf('Simulation ended.\n')

end

