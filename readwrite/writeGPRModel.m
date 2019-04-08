function  [] = writeGPRModel(model, filename)
%WRITESVMGRAD writes an svmgrad object to a text file
% o model   : gpr model object
% o filename: filename for text file
%
% The text file will follow the same order of variables
% * D:        Datapoint Dimension
% * N_train:  # of training data
% * x_train:  Training input   [DxN_train]
% * y_train:  Training output  [1xN_train]
% * Hyper-parameters 
% *   double length_scale; 
% *   double sigma_f; 
% *   double sigma_n; 
% This text file is to be read by the GPRwrap c++ class
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fileID = fopen(filename,'w');

% gpr_model.D
fprintf(fileID,'%d\n',model.D);
% gpr_model.N_train
fprintf(fileID,'%d\n',model.N_train);
% gpr_model.length_scale
fprintf(fileID,'%4.8f\n',model.length_scale);
% gpr_model.sigma_f
fprintf(fileID,'%4.8f\n\n',model.sigma_f);
% gpr_model.sigma_n
fprintf(fileID,'%4.8f\n\n',model.sigma_n);

% gpr_model.y_train
for i=1:model.N_train
    fprintf(fileID,'%4.8f ',model.y_train(i));
end
fprintf(fileID,'\n\n');

% gpr_model.x_train
for j=1:model.D
    for i=1:model.N_train
        fprintf(fileID,'%4.8f ',model.x_train(j,i));
    end
    fprintf(fileID,'\n');
end
fclose(fileID);

end

