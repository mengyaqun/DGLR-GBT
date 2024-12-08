
% purpose:  convert an image to the array of patches           **
% input:    im: target image                                   **
%           par: parameters                                    **
% output:   X: patch array                                     **
% Im2Patch函数的目的是将图像转换为 patches 数组。它接受一个目标图像im和一个参数par作为输入，并返回一个变量X，即 patches 数组。

function  X  =  Im2Patch( im, par )
f       =   par.win;
N       =   size(im,1)-f+1;
M       =   size(im,2)-f+1;
L       =   N*M;
X       =   zeros(f*f, L, 'single');
k       =   0;
for i  = 1:f
    for j  = 1:f
        k    =  k+1;
        blk  =  im(i:end-f+i,j:end-f+j);
        X(k,:) =  blk(:)';
    end
end
%初始化 patches 数组X，其尺寸为(f*f, L)，其中f是参数par中的窗口大小，L是 patches 数组的长度。
%初始化计数器k，用于记录 patches 数组中的当前 patch。
%使用嵌套循环遍历图像的每个像素。外循环控制行，内循环控制列。在每次迭代中，将当前像素值存储在临时变量blk中。
%然后将blk的转置（即按列存储）复制到 patches 数组X的第k行。
%在循环结束后，返回 patches 数组X。
%输出的变量X表示从输入图像im中提取的 patches 数组。