function [f,grad_f,hess_f] = gauss_functionwithGrad(x, Mu, Sigma)

f       = -my_gaussPDF(x, Mu, Sigma);

if nargout > 1 % gradient required
grad_f  = -grad_gauss_pdf(x, Mu, Sigma);
    if nargout > 2 % hessian required
        hess_f  = -hess_gauss_pdf(x, Mu, Sigma);
    end
end


end

