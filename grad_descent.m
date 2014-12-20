% 下载地址：http://www.mathworks.com/matlabcentral/fileexchange/35535-simplified-gradient-descent-optimization

function [xopt,fopt,niter,gnorm,dx] = grad_descent(varargin)

if nargin==0
    % define starting point
    x0 = [-1.5 2.9]';
elseif nargin==1
    % if a single input argument is provided, it is a user-defined starting
    % point.
    x0 = varargin{1};
else
    error('Incorrect number of input arguments.')
end

% termination tolerance
tol = -inf;

% maximum number of allowed iterations
maxiter = 1000;

% minimum allowed perturbation
dxmin = -inf;

% step size 
alpha = 1e-3;

% initialize gradient norm, optimization vector, iteration counter, perturbation
gnorm = inf; x = x0; niter = 0; dx = inf;

% define the objective function:
% demo function 正定二次函数
% f = @(x1,x2) x1.^2 + x1.*x2 + 3*x2.^2; 

% Rosenbrock function in 2 Dimension
f = @Rosenbrock;

% plot objective function contours for visualization:
x1 = -2:0.1:2;
x2 = -1:0.1:3;
[X1,X2]=meshgrid(x1,x2);
z =  f(X1,X2);
figure(1); clf;
n = 100; % the number of contour lines
contour(x1,x2,z,n);
hold on
plot(1,1,'rp')
hold on


% redefine objective function syntax for use with optimization:
f2 = @(x) f(x(1),x(2));

% gradient descent algorithm:
while and(gnorm>=tol, and(niter <= maxiter, dx >= dxmin))
    % calculate gradient:
    g = grad(x);
    gnorm = norm(g);
    % take step: 0.7e-1  or  2.6e-1
    [alpha,~] = fminbnd(@(alpha) iter(alpha,x,g),0,5e-2)
    xnew = x - alpha  *  g;
    % check step
    if ~isfinite(xnew)
        display(['Number of iterations: ' num2str(niter)])
        error('x is inf or NaN')
    end
    % plot current point
    h = plot([x(1) xnew(1)],[x(2) xnew(2)],'k.-');
    refreshdata(h,'caller');
    drawnow;
    hold on;
    % update termination metrics
    niter = niter + 1;
    dx = norm(xnew-x);
    x = xnew;
    
end
xopt = x;
fopt = f2(xopt);
niter = niter - 1;

end

% define the gradient of the objective
function g = grad(x)
g = [400*x(1).^3-400*x(1)*x(2)+2*x(1)-2
    200*x(2)-200*x(1).^2];
end

function frosen = iter(alpha,A,B)
    xnew = A - B*alpha;
    frosen = Rosenbrock(xnew(1),xnew(2));
end
