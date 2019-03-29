function [h_att_l, h_dirs] = plotLocalParams(att_l, local_basis, Mu)
[N,K] = size(att_l);
vel_size = 0.5;
if N == 2
    for n=1:N
        U = zeros(size(local_basis,3),1);
        V = zeros(size(local_basis,3),1);
        for i = 1:size(local_basis, 3)
            dir_    = local_basis(:,n,i)/norm(local_basis(:,n,i));
            U(i,1)   = dir_(1);
            V(i,1)   = dir_(2);
        end
        switch n
            case 1
                c = 'r';
            case 2
                c = 'g';
        end
        h_dirs = quiver(Mu(1,:)',Mu(2,:)', U, V, vel_size, 'Color', c, 'LineWidth',2); hold on;
    end
    h_att_l = []; h_atts_l = [];
    
    for k=1:K
        att_string = sprintf('$$\\xi^*_{%i}$$',k);
        h_atts_l = [h_atts_l scatter(att_l(1,k),att_l(2,k), 150, [1 0 0],'d','Linewidth',2)];
        h_att_l = [h_att_l text(att_l(1,k),att_l(2,k),att_string,'Interpreter', 'LaTex','FontSize',20, 'Color',[1 0 0])];
    end
end


end