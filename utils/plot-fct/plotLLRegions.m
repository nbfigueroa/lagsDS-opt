function [ h_gmm, h_ctr, h_txt ] = plotLLRegions( Mu, Sigma, varargin )
%UNTITLED10 Summary of this function goes here
%   Detailed explanation goes here
% Plot Gaussians on Data
[M, K] = size(Mu);
h_gmm = [];h_ctr = []; h_txt = [];

if nargin == 3
    choosen_active = varargin{1};
else
    choosen_active = 1:K;    
end
K_active = length(choosen_active);

if M==2
    for k=1:K_active
        jj = choosen_active(k);
        clust_color = [rand rand rand]  ;
        [h_gmm_, h_ctr_ ] = plotGMM(Mu(:,jj), Sigma(:,:,jj), clust_color, 1);
        h_gmm = [h_gmm h_gmm_];
%         h_ctr = [h_ctr h_ctr_];
        h_txt = [h_txt text(Mu(1,jj),Mu(2,jj),num2str(jj),'FontSize',20)];
    end
    box on
    grid on
    xlabel('$\xi_1$','Interpreter','LaTex','FontSize',20);
    ylabel('$\xi_2$','Interpreter','LaTex','FontSize',20);
%     colormap(hot)
    grid on
end

if M==3
    % Add the 3D option here...
end

end

