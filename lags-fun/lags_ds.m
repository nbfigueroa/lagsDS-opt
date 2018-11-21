function [x_dot] = lags_ds(attractor,x, mix_type, alpha_fun, A_c, b_c, A_t, b_t, varargin)
% Construct shift if given
T = eye(3); T_inv = eye(3);
if nargin > 8
    shift      = varargin{1} ;   
    T(1:2,3)   = shift;
    T_inv      = inv(T);    
    h_fun      = varargin{2};   
    A_s        = varargin{3};   
    b_s        = varargin{4}; 
    lambda_fun = varargin{5}; 
    grad_h_fun = varargin{6}; 
end

% Activation Region Separation
h = h_fun(x);

% Activation and Modulation functions
alpha = feval(alpha_fun,x)'; 
lambda = feval(lambda_fun,x);
grad_h = feval(grad_h_fun,x);
        
% Check incidence angle at local attractor
w = grad_h_fun(shift);
w_norm = -w/norm(w);
fg_att = A_c*shift + b_c;
fg_att_norm = fg_att/norm(fg_att);

% Put angles in nice range
angle_n  = atan2(fg_att_norm(2),fg_att_norm(1)) - atan2(w_norm(2),w_norm(1));
if(angle_n > pi)
    angle_n = -(2*pi-angle_n);
elseif(angle_n < -pi)
    angle_n = 2*pi+angle_n;
end

% Check if it's going against the grain
if angle_n > pi/2 || angle_n < -pi/2
    h_set = 0;
    corr_scale = 5;
%     corr_scale = 0.25;
else
    h_set = 1;
    corr_scale = 1;
end

% Check if deflection/correction is necessary
d_att = norm(shift-attractor);

% output velocity
x_dot       = zeros(size(x));
switch mix_type
    
    case 1
        alpha = feval(alpha_fun,x)';                           
        for i = 1:size(x,2)            
            % Compute Global Dynamics component        
            f_g = A_c*(x(:,i)- attractor); 
            
              
            % Compute Local Dynamics component                       
            if h(i) > 1 
                h_mod = 1;
            else                
                h_mod = h(i)*h_set;
            end
            
            % Compute Local Deflective Dynamics
            f_la = A_t*(x(:,i)-shift);
            f_ld = A_s*(x(:,i)-shift);
                        
            if d_att < 0.1
                f_l = h_mod*f_la;                 
                % Sum of two components            
                x_dot(:,i) = alpha(i)*f_g  + (1-alpha(i))*f_l ;
                
            else                
                % Compute Local Dynamics component
                f_l = (h_mod*A_t + (1-h_mod)*A_s)*(x(:,i)-shift); 
                
                % No modulation on the quasi-stable boundary
%                 x_dot(:,i) = alpha(i)*f_g  + (1-alpha(i))*f_l;
                
                % Sum of two components + correction           
                x_dot(:,i) = (alpha(i))*f_g  + (1 - alpha(i))*(f_l - corr_scale*lambda(i)* grad_h(:,i));
            end                                 
        end               
        
    case 2

        for i = 1:size(x,2)            
            % Compute Global Dynamics component
            f_g = A_c*(x(:,i)- attractor);                                   
                                                            
            % Compute Local Dynamics component                       
            if h(i) > 1 
                h_mod = 1;
            else                
                h_mod = h(i)*h_set;
            end
            
            % Compute Local Deflective Dynamics
            f_la = A_t*(x(:,i)-shift);
            f_ld = A_s*(x(:,i)-shift);
                                                     
            % Compute Local Dynamics component
            f_l = h_mod*f_la + (1-h_mod)*f_ld;                      
            
            % Sum of two components + correction         
            % No modulation on the quasi-stable boundary
            x_dot(:,i) = alpha(i)*f_g  + (1-alpha(i))*f_l;
            % With modulation on the quasi-stable boundary
%             x_dot(:,i) = alpha(i)*f_g  + (1-alpha(i))*(f_l - corr_scale*lambda(i)* grad_h(:,i));
        end
end

end