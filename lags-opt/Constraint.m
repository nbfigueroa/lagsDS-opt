function [f1, f2, f3, f4] = Constraint(x, varargin)

n = length(varargin);
f_h = cell(n,1);

for i = 1:n
   f_h{i} =  varargin{i};
end

if nargout > 2
    [f1_cell, f2_cell, f3_cell, f4_cell] = cellfun(@(c) c(x), f_h, 'UniformOutput',false);
    f1 = cat(1,f1_cell{:});
    f2 = cat(1,f2_cell{:});
    f3 = [f3_cell{:}];
    f4 = [f4_cell{:}];
else
    [f1_cell, f2_cell] = cellfun(@(c) c(x), f_h, 'UniformOutput',false);
    f1 = cat(1,f1_cell{:});
    f2 = cat(1,f2_cell{:});
end

end
