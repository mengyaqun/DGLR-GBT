
% purpose:  convert an image to the array of patches           **
% input:    im: target image                                   **
%           par: parameters                                    **
% output:   X: patch array                                     **
% Im2Patch������Ŀ���ǽ�ͼ��ת��Ϊ patches ���顣������һ��Ŀ��ͼ��im��һ������par��Ϊ���룬������һ������X���� patches ���顣

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
%��ʼ�� patches ����X����ߴ�Ϊ(f*f, L)������f�ǲ���par�еĴ��ڴ�С��L�� patches ����ĳ��ȡ�
%��ʼ��������k�����ڼ�¼ patches �����еĵ�ǰ patch��
%ʹ��Ƕ��ѭ������ͼ���ÿ�����ء���ѭ�������У���ѭ�������С���ÿ�ε����У�����ǰ����ֵ�洢����ʱ����blk�С�
%Ȼ��blk��ת�ã������д洢�����Ƶ� patches ����X�ĵ�k�С�
%��ѭ�������󣬷��� patches ����X��
%����ı���X��ʾ������ͼ��im����ȡ�� patches ���顣