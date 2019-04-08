function  [] = writeGPRTestData(x_test, y_test, filename)
%WRITESVMGRADTestData write testing data for the SVMGrad C++ implementation
% o x_test      : Dataset [DxM]
% o y_test      : labels  [1xM]
% o filename : -
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[D, M] = size(x_test);

fileID = fopen(filename,'w');

% D
fprintf(fileID,'%d\n',D);

% M
fprintf(fileID,'%d\n\n',M);

% x_test
for j=1:D
    for i=1:M
        fprintf(fileID,'%4.8f ',x_test(j,i));
    end
    fprintf(fileID,'\n');
end
fprintf(fileID,'\n');

% y_test
for i=1:M
    fprintf(fileID,'%4.8f ',y_test(i));
end
fprintf(fileID,'\n\n');

end

