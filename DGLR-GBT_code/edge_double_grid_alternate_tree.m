                                     
% Ŀ�ģ�    ����ͼ����ݶȺͱ�Եͼ 
% ���룺    X������ͼ�� 
%           nSig���������� 
% �����    BW_big��������ı�Եͼ 
%           T����Ե�����ֵ 
%           Y���ݶ� 

function [std_g, T, Y] = edge_double_grid_alternate_tree(X,nSig)
%edge_double_grid_alternate_tree��������һ����ά����X��һ������nSig��Ϊ���������
%����������������BW_big��T��Y���ú�����Ŀ����Ӧ��˫����������Ե����㷨����������X�У������ؽ����

X = double(X);  %����������Xת��Ϊ˫���ȡ�
[Nx,Ny]=size(X);    %��ȡ��������X��ά�ȣ�������洢��Nx��Ny��
Y = zeros(2*Nx,2*Ny);   %���������Y��ʼ��Ϊ�㡣

for i = 2:Nx-1  %ѭ��������������X���м��к��У����������ڵ�Ԫ���� x �� y �����ϵĲ�ֵ������ֵ�洢���������Y�С�
    for j = 2:Ny-1
        Y(2*(i-0.5)-1,2*j-1) = X(i,j)-X(i-1,j);
        Y(2*i-1,2*(j+0.5)-1) = X(i,j)-X(i,j+1);
    end
end

% First Column  ����������X�ĵ�һ�к����һ���Լ���һ�к����һ��ִ����ͬ�ļ��㡣
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
%��˫����������Ե����㷨Ӧ�����������Y��ͨ������������к��У������㵥Ԫ��ľ���ֵ��ƽ��ֵ�ͱ�׼�
% ���ƽ��ֵ�ӱ�׼��С�ڻ����nSig������ֵ����Ϊƽ��ֵ�ӱ�׼���nSig��Ȼ�󣬽��������BW_big��ֵ������ֵ�ĵ�Ԫ������Ϊ 1��
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
if sum(BW_big(:)) == 1  %����������BW_big�з��㵥Ԫ����������� 1�����������BW_big�е����е�Ԫ������Ϊ 0��
    BW_big = zeros(2*Nx, 2*Ny); 
end
