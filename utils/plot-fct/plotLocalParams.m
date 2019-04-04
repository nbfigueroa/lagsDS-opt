function [h_att_l, h_dirs] = plotLocalParams(att_l, local_basis, Mu, varargin)

[N,K] = size(att_l);
if nargin == 4
    choosen_active = varargin{1};
else
    choosen_active = 1:K;
end
K_active = length(choosen_active);

vel_size = 0.5;
if N == 2
    for n=1:N
        U = zeros(size(K_active,3),1);
        V = zeros(size(K_active,3),1);
        for i = 1:K_active
            dir_    = local_basis(:,n,choosen_active(i))/norm(local_basis(:,n,choosen_active(i)));
            U(i,1)   = dir_(1);
            V(i,1)   = dir_(2);
        end
        switch n
            case 1
                c = 'r';
            case 2
                c = 'g';
        end
        Mu_ = Mu(:,choosen_active);        
        h_dirs = quiver(Mu_(1,:)',Mu_(2,:)', U, V, vel_size, 'Color', c, 'LineWidth',2); hold on;
    end
    h_att_l = []; h_atts_l = [];
    
    for i=1:K_active
        k = choosen_active(i);
        att_string = sprintf('$$\\xi^*_{%i}$$',k);
        h_atts_l = [h_atts_l scatter(att_l(1,k),att_l(2,k), 150, [1 0 0],'d','Linewidth',2)];
        h_att_l = [h_att_l text(att_l(1,k),att_l(2,k),att_string,'Interpreter', 'LaTex','FontSize',20, 'Color',[1 0 0])];
    end
end


end