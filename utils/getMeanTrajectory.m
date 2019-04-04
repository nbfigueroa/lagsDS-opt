function [Data_mean] = getMeanTrajectory(data, att_g, sub_sample, dt)
N = size(data{1},1)/2
nDemos  = length(data);
data_g = [];
data_g = data;
for i=1:nDemos
    data_g{i}(1:N,:) = data{i}(1:N,:) + repmat(att_g,[1 length(data{i})]);
end

[~, target_trajectory] = processDrawnDataset(data_g, 1:nDemos, sub_sample, dt);
fprintf('done\n');

% Uses the mean of all demonstrations as target_trajectory Xi
[dim, lX] = size(target_trajectory);
target_trajectory = target_trajectory./nDemos;
target_trajectoryV = [diff(target_trajectory,[],2),zeros(dim,1)]./dt;
Data_mean = [target_trajectory;target_trajectoryV];
Data_mean = Data_mean(:,1:sub_sample:end);
end