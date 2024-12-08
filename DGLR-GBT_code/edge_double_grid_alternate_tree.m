                                     
% 目的：    计算图像的梯度和边缘图 
% 输入：    X：噪声图像 
%           nSig：噪声方差 
% 输出：    BW_big：计算出的边缘图 
%           T：边缘检测阈值 
%           Y：梯度 

function [std_g, T, Y] = edge_double_grid_alternate_tree(X,nSig)
%edge_double_grid_alternate_tree函数接受一个二维数组X和一个整数nSig作为输入参数，
%并返回三个变量：BW_big、T和Y。该函数的目的是应用双网格交替树边缘检测算法到输入数组X中，并返回结果。

X = double(X);  %将输入数组X转换为双精度。
[Nx,Ny]=size(X);    %获取输入数组X的维度，并将其存储在Nx和Ny中
Y = zeros(2*Nx,2*Ny);   %将输出数组Y初始化为零。

for i = 2:Nx-1  %循环遍历输入数组X的中间行和列，并计算相邻单元格在 x 和 y 方向上的差值。将差值存储在输出数组Y中。
    for j = 2:Ny-1
        Y(2*(i-0.5)-1,2*j-1) = X(i,j)-X(i-1,j);
        Y(2*i-1,2*(j+0.5)-1) = X(i,j)-X(i,j+1);
    end
end

% First Column  对输入数组X的第一行和最后一行以及第一列和最后一列执行相同的计算。
for i = 2:Nx-1
    j = 1;
    
    Y(2*(i-0.5)-1,2*j-1) = X(i,j)-X(i-1,j);
    Y(2*i-1,2*(j+0.5)-1) = X(i,j)-X(i,j+1);
end

% Last Column
for i = 2:Nx
    j = Ny;
    
    Y(2*(i-0.5)-1,2*j-1) = X(i,j)-X(i-1,j);
end

% First Row
for j = 1:Ny-1
    i = 1;    
    Y(2*i-1,2*(j+0.5)-1) = X(i,j)-X(i,j+1);
end

% Last Row
for j = 1:Ny-1
    i = Nx;   
    Y(2*(i-0.5)-1,2*j-1) = X(i,j)-X(i-1,j);
    Y(2*i-1,2*(j+0.5)-1) = X(i,j)-X(i,j+1);
end
%将双网格交替树边缘检测算法应用于输出数组Y。通过遍历数组的行和列，并计算单元格的绝对值的平均值和标准差。
% 如果平均值加标准差小于或等于nSig，则将阈值设置为平均值加标准差加nSig。然后，将输出数组BW_big中值大于阈值的单元格设置为 1。
[ifull,jfull] = meshgrid(1:2*Nx,1:2*Ny);
ifull = ifull(:); jfull = jfull(:);
ixkeep = find((mod(ifull,2) == 0) | (mod(jfull,2) == 0));

mu_g = mean(abs(Y(ixkeep)));
std_g = std(abs(Y(ixkeep)));
T = (mu_g + std_g);  
if T <= nSig
    T = (mu_g + std_g+nSig);
end
    
BW_big = zeros(2*Nx, 2*Ny);
BW_big(abs(Y)>T) = 1;
if sum(BW_big(:)) == 1  %如果输出数组BW_big中非零单元格的总数等于 1，则将输出数组BW_big中的所有单元格设置为 0。
    BW_big = zeros(2*Nx, 2*Ny); 
end
