function X = DUAL_ADMM(params)
%=============================================
% Dual Graph regularized sparse coding algorithm (DUAL_ADMM).
%
% DUAL_ADMM solves the optimization problem:
%       min  |Y-D*X|_F^2 + beta*tr(X*Lc*X')+alpha*tr(X'*Lr*X) s.t.  |x_i|_0 <= T
%        X  
% Main parameters:
%       D - the dictionary (its columns MUST be normalized)
%       Y - the signals to represent
%       T - sparsity constraint (max. number of coefficients per signal)
%       L - manifold graph Laplacian
%       beta - regularization coefficient
%       X - initial sparse code (default: run non-regularized OMP)
%       iternum - number of ADMM iterations (default: 25)
%       rho - ADMM step size parameter (default: 1)
%       runDebiasing - update values on the determined support using least-squares (default: 1)
%       rundebias -
% Output:
%       X - sparse coefficient matrix
%{

%}
if ~exist('omp','file')
    error('OMP Package missing!');
end

if isfield(params,'Y')
    Y = params.Y;
else
    error('Input data matrix Y missing!');
end

if isfield(params,'Z')
    Z = params.Z;
else
    error('Input data matrix Z missing!');
end

if isfield(params,'D')
    D = params.D;
else
    error('Input dictionary D missing!');
end

if isfield(params,'T')
    T = params.T;
else
    error('Sparsity constraint T missing!');
end

if isfield(params,'Lc')
    Lc = params.Lc;
else
    error('Manifold Laplacian Lc missing!');
end

if isfield(params,'Lr')
    Lr = params.Lr;
else
    error('Manifold Laplacian Lr missing!');
end

if isfield(params,'alpha')
    alpha = params.alpha;
else
    error('Regularizaion coefficient alpha missing!');
end

if isfield(params,'beta')
    beta = params.beta;
else
    error('Regularizaion coefficient beta missing!');
end

if isfield(params,'iternum')
    iternum = params.iternum;
else
    iternum = 25;
end

if isfield(params,'rho')
    rho = params.rho;
else
    rho = 1;
end

M = size(Y,2); % number of signals
K = size(D,2); % number of atoms
U = zeros(K,M);
Z = U;
A=D'*D+rho*eye(size(D'*D))+alpha*Lr;
B=beta*Lc;
C=D'*Y+rho*(Z-U);
%ADMM
for i = 1:iternum 
    %X = sylvester(full(D'*D+rho*eye(size(D'*D))+alpha*Lr),full(beta*Lc),full(D'*Y+rho*(Z-U)));
    X = sylvester(full(A),full(B),full(C));
    Z = SpProj(X+U,T);
    U = U+X-Z;
end
%{ update values on the determined support using LS
end


%{

%}
function Z = SpProj(XU,T)
Z = zeros(size(XU));
for j=1:size(Z,2)
    [~,ind] = sort(abs(XU(:,j)),'descend');
    Z(ind(1:T),j) = XU(ind(1:T),j);
end
end
