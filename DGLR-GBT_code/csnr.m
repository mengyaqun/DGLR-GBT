
% purpose:  compute CSNR                                       

function [ s ] = csnr(A,B,row,col)
%函数 csnr 接收两个输入数组 A 和 B，以及两个整数值 row 和 col，并返回一个变量 s。该函数计算输入数组 A 和 B 之间的对比噪声比 (CSNR)。
[n,m,ch]=size(A);

if ch==1
   e=A-B;
   e=e(row+1:n-row,col+1:m-col);
   me=mean(mean(e.^2));
   s=10*log10(255^2/me);
else
   e=A-B;
   e=e(row+1:n-row,col+1:m-col,:);
   e1=e(:,:,1);e2=e(:,:,2);e3=e(:,:,3);
   me1=mean(mean(e1.^2));
   me2=mean(mean(e2.^2));
   me3=mean(mean(e3.^2));
   mse=(me1+me2+me3)/3;
   s  = 10*log10(255^2/mse);
end
%获取输入数组 A的尺寸，并将其存储在 n、m 和 ch 中。检查输入数组 A 中的通道数 (ch) 是否为 1。否则，继续下一步。
%计算输入数组 A 和 B 的差值 (e)，并将差值数组的尺寸限制在指定的行和列值范围内。
%计算差值数组有限维数的平方差平均值 (me)。
%通过平均每个通道的平方差均值计算均方误差 (mse)。
%将对比度 (255^2/mse) 的对数乘以 10，计算 CSNR。
%将计算出的 CSNR 值存储在变量 s 中并返回。
%输出变量 s 表示输入阵列 A 和 B 之间的对比度-噪声比。
