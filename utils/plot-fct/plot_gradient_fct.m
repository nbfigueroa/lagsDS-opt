function [h] = plot_gradient_fct(grad_fun, limits, title_string)

% nx = 50; ny = 50;
nx = 40; ny = 40;
axlim = limits;
ax_x=linspace(axlim(1)*0.99,axlim(2)*0.99,nx); % computing the mesh points along each axis
ax_y=linspace(axlim(3)*0.99,axlim(4)*0.99,ny); % computing the mesh points along each axis
[x_tmp, y_tmp]=meshgrid(ax_x,ax_y);  % meshing the input domain
x=[x_tmp(:), y_tmp(:)]';
grad_eval = feval(grad_fun,x);
U = zeros(size(grad_eval,2),1);
V = zeros(size(grad_eval,2),1);
for i = 1:size(grad_eval, 2)    
   gradient = grad_eval(:,i);
   gradient = gradient/norm(gradient);
   U(i,1)   = gradient(1);
   V(i,1)   = gradient(2);
end
h = quiver(x(1,:),x(2,:), U', V', 0.75,  'Color', 'k', 'LineWidth',1);
title(title_string, 'Interpreter','LaTex','FontSize', 18);

end