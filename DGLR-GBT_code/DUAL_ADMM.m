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
% 双图正则化稀疏编码算法 (DUAL_ADMM)。
% DUAL_ADMM 解决优化问题：
min |Y-D*X|_F^2 + beta*tr(X*Lc*X')+alpha*tr(X'*Lr*X) s.t. |x_i|_0 <= T
% X  
% 主要参数：
D - 字典（其列必须标准化）
% Y - 要表示的信号
% T--稀疏性约束（每个信号的最大系数数）
% L - 流形图拉普拉卡方
% beta - 正则化系数
% X - 初始稀疏代码（默认：运行非正则化 OMP）
% iternum - ADMM 的迭代次数（默认值：25）
% rho - ADMM 步长参数（默认值：1）
% runDebiasing - 使用最小二乘法更新已确定的支持值（默认：1）
% rundebias
% 输出：
% X - 稀疏系数矩阵
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
在 MATLAB 中，这段代码定义了一个名为SpProj的函数，它接受两个参数：
XU（输入矩阵）和T（目标稀疏度）。函数的主要目的是对输入矩阵进行稀疏投影。
首先，创建一个与输入矩阵XU大小相同的零矩阵Z，用于存储稀疏投影结果。
接下来，使用一个 for 循环遍历矩阵Z的列。在循环内部，首先计算输入矩阵XU的第j列的绝对值，并按降序排序。
然后，将排序后的索引ind分成两部分：ind(1:T) 和 ind(T+1:end)。前T个最大绝对值的元素对应于矩阵Z的第j列的前T个元素，
即 Z(ind(1:T),j)。最后，将XU中对应这些索引的元素赋值给Z的第j列，完成稀疏投影。
这样，SpProj函数就实现了对输入矩阵XU的稀疏投影，并将结果存储在矩阵Z中。
%}
function Z = SpProj(XU,T)
Z = zeros(size(XU));
for j=1:size(Z,2)
    [~,ind] = sort(abs(XU(:,j)),'descend');
    Z(ind(1:T),j) = XU(ind(1:T),j);
end
end
