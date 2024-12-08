
% purpose:  compute weighted GBT basis functions               **
%           for a patch with AWGN                              **
% input:    patch: noisy patch                                 **
%           nSig: noise variance                               **
% output:   V: computed GBT basis                              **
%WGBT������Ŀ��:�����Ȩ���屴Ҷ˹�任��weighted GBT�������������ڴ�����м��Ը�˹��������AWGN����ͼ��顣
%�ú�������һ������ͼ���patch��һ����������nSig��Ϊ���������������һ������V��������õ��� GBT ��������

function [ V ] = WGBT(patch,nSig)  

%1-����������ͼ��   ���ȵ���edge_double_grid_alternate_tree��������������ͼ���patch����������nSig��
%�õ���Ե���ͼ��BW_big����ֵT���ݶ�gd��Ȼ������Ȩ�����Ծ���SimMat��ʹ���ݶ�gd��ƽ������һ��Ԥ���趨����ֵsigma��
[std_g, T, gd] = edge_double_grid_alternate_tree(patch,nSig);  % gd: gradient
alpha = 0.2;  % 0.1~0.2
sigma = alpha*(nSig + max(max(abs(gd))));
%SimMat = exp(-(gd.^2)/(sigma^2));
SimMat = exp(-(gd.^2+(std_g^2))/(sigma^2));		%����
%2-������������������ɼ�Ȩ�ڽӾ���W��Ȼ�����Խ��߾���D�����ż�����Ⱦ���L�����ͨ������ֵ�ֽ⣨SVD�����������V��Ȩ��Lam��
W = AdjMatWei(SimMat);  % generate weighted adjacency matrix  
D = diag(sum(W,2));  
L = D - W;
[V, Lam] = eig(L);
end

