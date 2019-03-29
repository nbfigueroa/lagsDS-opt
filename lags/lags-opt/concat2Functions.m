function [out1, out2] = concat2Functions(x, f1, f2)
out1 = feval(f1,x);
out2 = feval(f2,x);
end

