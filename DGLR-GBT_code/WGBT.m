
% purpose:  compute weighted GBT basis functions               **
%           for a patch with AWGN                              **
% input:    patch: noisy patch                                 **
%           nSig: noise variance                               **
% output:   V: computed GBT basis                              **
%WGBT函数的目的:计算加权广义贝叶斯变换（weighted GBT）基函数，用于处理带有加性高斯白噪声（AWGN）的图像块。
%该函数接受一个噪声图像块patch和一个噪声方差nSig作为输入参数，并返回一个变量V，即计算得到的 GBT 基函数。

function [ V ] = WGBT(patch,nSig)  

%1-构建相似性图。   首先调用edge_double_grid_alternate_tree函数，输入噪声图像块patch和噪声方差nSig，
%得到边缘检测图像BW_big、阈值T和梯度gd。然后计算加权相似性矩阵SimMat，使用梯度gd的平方除以一个预先设定的阈值sigma。
[std_g, T, gd] = edge_double_grid_alternate_tree(patch,nSig);  % gd: gradient
alpha = 0.2;  % 0.1~0.2
sigma = alpha*(nSig + max(max(abs(gd))));
%SimMat = exp(-(gd.^2)/(sigma^2));
SimMat = exp(-(gd.^2+(std_g^2))/(sigma^2));		%修正
%2-计算基向量。首先生成加权邻接矩阵W，然后计算对角线矩阵D，接着计算低秩矩阵L。最后，通过奇异值分解（SVD）计算基函数V和权重Lam。
W = AdjMatWei(SimMat);  % generate weighted adjacency matrix  
D = diag(sum(W,2));  
L = D - W;
[V, Lam] = eig(L);
end

