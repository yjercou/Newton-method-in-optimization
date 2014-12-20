% 2014/12/20
% by Y Jay

function [xopt,fopt,niter,gnorm,dx] = newton_opt(varargin)

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
tol = 1e-5;

% maximum number of allowed iterations
maxiter = 100;

% minimum allowed perturbation
dxmin = 1e-5;

% initialize gradient norm, optimization vector, iteration counter, perturbation
gnorm = inf; x = x0; niter = 0; dx = inf;

% define the objective function:
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
    % take step: by Hesse and grad
    hes= hesse(x);
    [lamda,~] = fminbnd(@(lamda) iter(lamda,hes,x,g),0,10)
    
    xnew = x - lamda * hes\g;
    
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

function hes = hesse(x)
hes = [1200*x(1)^2-400*x(2)+2, -400*x(1);
    -400*x(1), 200];
end

function frosen = iter(lamda,hes,A,B)
    xnew = A - lamda*hes\B;
    frosen = Rosenbrock(xnew(1),xnew(2));
end
