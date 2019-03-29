function [out1, out2, out3] = concat3Functions(x, f1, f2,f3)
out1 = feval(f1,x);
out2 = feval(f2,x);
out3 = feval(f3,x);
end

