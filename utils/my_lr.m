function [ w, y_est ] =  my_lr(X, y)
%MY_LR Implementation of the Least-Squares estimation for a linear
%   regression model.
%
%   input -----------------------------------------------------------------
%   
%       o X       : (M x N), a data set with M samples each being of 
%                            dimension N each column corresponds to a datapoint
%       o y       : (M x 1), a vector with outputs y \in R corresponding to X.
%
%   output ----------------------------------------------------------------
%
%       o w       : (N x 1), a vector with estimated linear coefficients of 
%                           of the regressive model y = w'X
%       o y_est   : (M x 1), a vector with estimated outputs y_hat 
%                   corresponding to X.
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Least-Squares Regression (M x N) : Works woth 1D data in same format
w = (X'*X)^-1 * (X'*y);
% w = (X'*y)/(X'*X);

% Linear Regressive Function (M x 1)
y_est  = X*w;

%%%%% Differences in Literature: ABOVE is standard way
%%%%% below is with transposed matrices. i.e. X \in (N x M) and y (1 x M)%%
% Least-Squares Regression (N x M)
% w = (X*X')^-1 * X * y';

% Linear Regressive Function (1 x M)
% y_est  = w' * X;

end