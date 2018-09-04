function [h] = plot_lyap_fct(lyap_fun, contour, limits, title_string, min_max)

% Plot the Lyapunov Function Contours
nx = 200; ny = 200;
axlim = limits;
ax_x=linspace(axlim(1),axlim(2),nx); % computing the mesh points along each axis
ax_y=linspace(axlim(3),axlim(4),ny); % computing the mesh points along each axis
[x_tmp, y_tmp]=meshgrid(ax_x,ax_y);  % meshing the input domain
x=[x_tmp(:), y_tmp(:)]';
[ys2] = feval(lyap_fun,x);
z_tmp = reshape(ys2,nx,ny);
if contour
    h = figure('Color',[1 1 1]); hc = contourf(x_tmp,y_tmp,z_tmp,40);
%     set(hc,'LineWidth',1)
else
    h = figure('Color',[1 1 1]); surfc(x_tmp,y_tmp,z_tmp); 
    shading interp; alpha 0.5;
end

level = 200; n = ceil(level/2);
cmap1 = [linspace(1, 1, n); linspace(0, 1, n); linspace(0, 1, n)]';
cmap2 = [linspace(1, 0, n); linspace(1, 0, n); linspace(1, 1, n)]';
cmap =  [cmap1; cmap2(2:end, :)];
colormap(vivid(cmap, [.5, .5]));
colorbar

if ~isempty(min_max)
    switch min_max
        case 0
            [minimum_vdot, min_id_vdot]  = min(z_tmp(:));
            if contour
                minimum_vdot = 0;
            end
            hold on; scatter3(x(1,min_id_vdot),x(2,min_id_vdot),minimum_vdot,50,[1 0 0],'filled');
            x_min = [x(1,min_id_vdot),x(2,min_id_vdot)];
            fprintf('Minimum value: %3.4f at [%3.4f,%3.4f]\n',minimum_vdot,x_min(1),x_min(2))
        case 1
            [maximum_vdot, max_id_vdot]  = max(z_tmp(:));
            if contour
                maximum_vdot = 0;
            end
            hold on; scatter3(x(1,max_id_vdot),x(2,max_id_vdot),maximum_vdot,50,[1 0 0],'filled');
            x_max = [x(1,max_id_vdot),x(2,max_id_vdot)];
            fprintf('Maximum value: %3.4f at [%3.4f,%3.4f]\n',maximum_vdot,x_max(1),x_max(2))
    end
end
title(title_string, 'Interpreter','LaTex','FontSize', 18);

end