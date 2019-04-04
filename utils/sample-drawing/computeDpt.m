function dpt = computeDpt(r1, r2, theta)
dpt_sin = (r1*sin(theta))^2;
dpt_cos = (r1*cos(theta))^2;
dpt  = sqrt(dpt_sin + dpt_cos);
end