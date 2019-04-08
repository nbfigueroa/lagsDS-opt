%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%        Data Processing Script      %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; clc; close all
pkg_dir         = '/home/nbfigueroa/Dropbox/PhD_papers/LAGS-paper/new-code/lagsDS-opt/';
% load(strcat(pkg_dir,'datasets/2d-locomotion/rawdata_icub_narrow'))
load(strcat(pkg_dir,'datasets/2d-locomotion/rawdata_object_conveyor'))

data = []; 
odata = [];
window_size = 751; crop_size = (window_size+1)/2; 
dt = mean(abs(diff(raw_data{1}(1,:))));

raw_data(1) = [];

for i=1:length(raw_data)
        % Cut off the first 500 messages
        if i == 1
            raw_data{i} = raw_data{i}(:,2000:end);
        else
            raw_data{i} = raw_data{i}(:,1000:end);
        end
        raw_data{i}(:,end-50:end) = [];
    
        % Smooth position and compute velocities
        dx_nth = sgolay_time_derivatives(raw_data{i}(2:3,:)', dt, 2, 2, window_size);
        X     = dx_nth(:,:,1)';
        X_dot = dx_nth(:,:,2)';               
        theta_angles = raw_data{i}(4,crop_size:end-crop_size);
        
        if i == 4
            X(:,1:700) = [];
            X_dot(:,1:700) = [];
            theta_angles(:,1:700) = [];
        elseif i > 4
            X(:,1:1000) = [];
            X_dot(:,1:1000) = [];
            theta_angles(:,1:1000) = [];
        end
        
        
        data{i} = [X; X_dot];                
        odata{i} = theta_angles;
        
        % Compute rotation data in different forms
        R = zeros(3,3,length(theta_angles));
        H = zeros(4,4,length(theta_angles));
        for r=1:length(theta_angles)
            % Populate R matrix
            R(:,:,r)  = eul2rotm([theta_angles(r),0,0]');
            
            % Populate H matrix
            H(:,:,r)     = eye(4);
            H(1:3,1:3,r) = R(:,:,r);
            H(1:3,4,r)   = [data{i}(1:2,r); 0] ;
            
        end
        q = quaternion(R,1);
        Rdata{i} = R;
        Hdata{i} = H;
        qdata{i} = q;                
end

% Trajectories to use
left_traj = 1;
if ~left_traj
    data(4:end) = [];
    odata(4:end) = [];
    qdata(4:end) = [];
    Rdata(4:end) = [];
    Hdata(4:end) = [];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Sub-sample measurements and Process for Learning      %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sub_sample = 5;
[Data, Data_sh, att, x0_all, ~, data, Hdata] = processDataStructureOrient(data, Hdata, sub_sample);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Visualize 2D reference trajectories %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Position/Velocity Trajectories
vel_samples = 50; vel_size = 0.5; 
[h_data, h_att, h_vel] = plot_reference_trajectories_DS(Data, att, vel_samples, vel_size);
axis equal

% Draw Obstacles for Narrow Passage setting
% rectangle('Position',[-1.5 1 3 2], 'FaceColor',[.85 .85 .85 0.3]); hold on;
% rectangle('Position',[2 1 3 2], 'FaceColor',[.85 .85 .85 0.3]); hold on;

% Draw Obstacles for Conveyor Belt Setting
rectangle('Position',[-0.3 -1.5 0.6 3], 'FaceColor',[.85 .85 .85 0.3]); hold on;
rectangle('Position',[2.5 3.7 3 0.6], 'FaceColor',[.85 .85 .85 0.3]); hold on;
h_att = scatter(att(1),att(2), 150, [0 0 0],'d','Linewidth',2); hold on;


%%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Visualize 6DoF data in 3d %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extract Position and Velocities
M          = size(Data,1)/2;    
Xi_ref     = Data(1:M,:);
Xi_dot_ref = Data(M+1:end,:);   

% Make 3D
M = 3;
Xi_ref(M,:)     = 0;
Xi_dot_ref(M,:) = 0;
Data_xi = [Xi_ref; Xi_dot_ref];
att_xi = att; att_xi(M,1) = 0;

%%%%% Plot 3D Position/Velocity Trajectories %%%%%
vel_samples = 50; vel_size = 0.75; 
[h_data, h_att, h_vel] = plot_reference_trajectories_DS(Data_xi, att_xi, vel_samples, vel_size); 
hold on;

%%%%%% Plot Wall %%%%%%
% cornerpoints = [-1 1 0;  5 1 0; 5 2 0; -1 2 0;
%                 -1 1 0.25;  5 1 0.25; 5 2 0.25; -1 2 0.25];            
% plotminbox(cornerpoints,[0.5 0.5 0.5]); hold on;


%%%%% Plot 6DoF trajectories %%%%%
ori_samples = 300; frame_size = 0.25; box_size = [0.45 0.15 0.05];
plot_6DOF_reference_trajectories(Hdata, ori_samples, frame_size, box_size,'r'); 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Playing around with quaternions   %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Quaternions
figure('Color',[1 1 1])
for i=1:length(qdata)  
    qData = qdata{i};
    plot(1:length(qData),qData(1,:),'r-.','LineWidth',2); hold on;
    plot(1:length(qData),qData(2,:),'g-.','LineWidth',2); hold on;
    plot(1:length(qData),qData(3,:),'b-.','LineWidth',2); hold on;
    plot(1:length(qData),qData(4,:),'m-.','LineWidth',2); hold on;
    legend({'$q_1$','$q_2$','$q_3$','$q_4$'},'Interpreter','LaTex', 'FontSize',14)
    xlabel('Time-stamp','Interpreter','LaTex', 'FontSize',14);
    ylabel('Reference Quaternions','Interpreter','LaTex', 'FontSize',14);    
    grid on;
    axis tight;
end
title('Demonstrations from Gazebo','Interpreter','LaTex', 'FontSize',14);

%% Compute Omega from qdata
Odata = [];
for i=1:length(qdata)   
    qData = qdata{i}; 
    RData = Rdata{i};
    Omega = zeros(3,length(qData));
    for ii=2:length(qData)
        if true
            q_2 = quat_conj(qData(:,ii-1));
            q_1 = qData(:,ii);
            
            % Partitioned product
            % delta_q = quat_prod(q_1,q_2);            
            % Matrix product option 1
            % Q = QuatMatrix(q_1);
            % delta_q = Q*q_2;
            
            % Matrix product option 2
            delta_q = quat_multiply(q_1',q_2');            
            Omega(:,ii) = 2*quat_logarithm(delta_q)/dt;
        else
            % Using Rotation matrices
            R_2 = RData(:,:,ii-1);
            R_1 = RData(:,:,ii);
            Omega(:,ii) = rot_logarithm(R_1*R_2');
        end
    end                
    Odata{i} = Omega;
end

% Plot Angular Velocities
figure('Color',[1 1 1])
% for i=1:length(Odata)   
for i=1:1
    Omega = Odata{i};            
    plot(1:length(Omega),Omega(1,:),'r-.','LineWidth',2); hold on;
    plot(1:length(Omega),Omega(2,:),'g-.','LineWidth',2); hold on;
    plot(1:length(Omega),Omega(3,:),'b-.','LineWidth',2); hold on;
    legend({'$\omega_1$','$\omega_2$','$\omega_3$'},'Interpreter','LaTex', 'FontSize',14)
    xlabel('Time-stamp','Interpreter','LaTex', 'FontSize',14);
    ylabel('Angular Velocity (rad/s)','Interpreter','LaTex', 'FontSize',14);    
    grid on;
    axis tight;
end
title('Demonstrations from Gazebo','Interpreter','LaTex', 'FontSize',14);

%% Plot Integrated Quaternions
figure('Color',[1 1 1])
for i=1:length(Odata)   
    Omega = Odata{i};
    qData = qdata{i};
    qData_hat = zeros(4,length(Omega));    
    
    % Forward integration
    qData_hat(:,1) = qData(:,1); 
    for ii=2:length(qData_hat)
        omega_exp = quat_exponential(Omega(:,ii), dt);
        qData_hat(:,ii) = real(quat_multiply(omega_exp',qData_hat(:,ii-1)'));
    end
    
    % Plot Forward integrated quaternions
    plot(1:length(qData_hat),qData_hat(1,:),'r-.','LineWidth',2); hold on;
    plot(1:length(qData_hat),qData_hat(2,:),'g-.','LineWidth',2); hold on;
    plot(1:length(qData_hat),qData_hat(3,:),'b-.','LineWidth',2); hold on;
    plot(1:length(qData_hat),qData_hat(4,:),'m-.','LineWidth',2); hold on;
    legend({'$\hat{q}_1$','$\hat{q}_2$','$\hat{q}_3$','$\hat{q}_4$'},'Interpreter','LaTex', 'FontSize',14)
    xlabel('Time-stamp','Interpreter','LaTex', 'FontSize',14);
    ylabel('Forward Inegrated Quaternion $q(t + \Delta t)$','Interpreter','LaTex', 'FontSize',14);    
    grid on;
    axis tight;
end
title('Demonstrations from Gazebo','Interpreter','LaTex', 'FontSize',14);



