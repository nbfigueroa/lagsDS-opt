function [Data, Data_sh, att, x0_all, dt, data, qdata, Hdata, Data_QX, dataset_name, box_size] = load_loco_datasets(pkg_dir, choosen_dataset, sub_sample)

switch  choosen_dataset
    case 1             
             load(strcat(pkg_dir,'datasets/2d-locomotion/procdata_icubGazebo_right'))
             dataset_name = 'Gazebo Demonstrations-1';
             box_size = [0.45 0.15 0.05];
             for i=1:3
                 data{i} = data{i}(:,400:end);
                 qdata{i} = qdata{i}(:,400:end);
             end             

    case 2             
             load(strcat(pkg_dir,'datasets/2d-locomotion/procdata_icubGazebo_full'))
             dataset_name = 'Gazebo Demonstrations-2';
             box_size = [0.45 0.15 0.05];  
             for i=1:3
                 data{i} = data{i}(:,400:end);
                 qdata{i} = qdata{i}(:,400:end);
             end
             
    case 3        
            load(strcat(pkg_dir,'datasets/2d-locomotion/procdata_icubwMichael'))
            dataset_name = 'Real Demonstrations w/Mike';
            box_size = [0.15 0.1 0.05];
            
    case 4        
            load(strcat(pkg_dir,'datasets/2d-locomotion/procdata_icub_narrow'))
            dataset_name = 'iCub Narrow Passage Walking';
            box_size = [0.15 0.1 0.05];            
            data(3) = [];
            qdata(3) = [];
            
    case 5        
            load(strcat(pkg_dir,'datasets/2d-locomotion/procdata_object_conveyor'))
            dataset_name = 'iCubs Object Conveyor Co-Manipulation';
            box_size = [0.15 0.1 0.05];                        
end

% Process and sub-sample trajectories... might need to add qdata
[Data, Data_sh, att, x0_all, ~, data, Hdata] = processDataStructureOrient(data, Hdata, sub_sample);

% Create dataset for learning joint distribution (Quaternion/Xi)
Data_QX = [];
for i=1:length(data)
    q_data     = qdata{i}(:,1:sub_sample:end);
    xi_data    = data{i}(1:2,1:sub_sample:end);    
    q_xi       = [q_data; xi_data];
    Data_QX    = [Data_QX q_xi];  
end


end