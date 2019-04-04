function pnts = draw_from_isocontour( Sigma, Mu, npts, c )

% Simple numerical scheme to draw samples from 

% get number of dimensions of the covariance matrix
ndims = length(Sigma);

% calculate eigenvalues and vectors of the covariance matrix
[V, L] = eig(Sigma);

% check size of Mu and transpose if necessary
sc = size(Mu);
if sc(1) > 1
    Mu = Mu';
end 

% Generate radii of desired hyper-ellipsoid
sigma1 = L(1,1); sigma2 = L(2,2);
sigma_cte = sqrt(sigma1)*sqrt(sigma2);
r1 = sqrt(2* sigma1 * log(1/(2*pi*c*sigma_cte)));
r2 = sqrt(2* sigma2 * log(1/(2*pi*c*sigma_cte)));

% Variables for Integral Sampling on the isocontour
theta = 0; delta_Theta = 0.0001; circ = 0.0; 
num_Integrals = round((2*pi)/delta_Theta);

% integrate over the ellipse to get the circumference
for i=0:num_Integrals-1
    theta = theta + i*delta_Theta;
    dpt = computeDpt(r1, r2, theta);
    circ = circ + dpt;
end
% fprintf('Circumference %2.2f\n',circ);

% Sample points on perimeter
run = 0; nextPoint = 0;
theta = 0;
pnts = [];
for i=1:num_Integrals
    theta = theta + delta_Theta;
    if ((npts*run/circ) >= nextPoint)
        xy = [r1 * cos(theta) r2 * sin(theta)];
        pnts = [pnts; xy];
        nextPoint = nextPoint + 1; 
    end
    run = run + computeDpt(r1, r2, theta);
end

for i=1:length(pnts)
    
    % Rotate and translate ellipsoid
    pnts(i,:) = pnts(i,:)* V' + Mu;
end
pnts = pnts';

end

