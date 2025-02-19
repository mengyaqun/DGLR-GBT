clc;
clear;

PSNR=zeros(30);
SSIM=zeros(30);
for v=1:18
    if v==1
        im_in = double(imread('D:\image desioning\MATLAB\RGBZ\1100.png'));
    elseif v==2
        im_in = double(imread('D:\image desioning\MATLAB\RGBZ\1110.png'));  % the input original image
    elseif v==3
        im_in = double(imread('D:\image desioning\MATLAB\RGBZ\1120.png'));
    elseif v==4 
        im_in = double(imread('D:\image desioning\MATLAB\RGBZ\1130.png'));
    elseif v==5
        im_in = double(imread('D:\image desioning\MATLAB\RGBZ\1140.png'));
    elseif v==6
        im_in = double(imread('D:\image desioning\MATLAB\RGBZ\1150.png'));
    elseif v==7
        im_in = double(imread('D:\image desioning\MATLAB\RGBZ\1160.png'));
    elseif v==8
        im_in = double(imread('D:\image desioning\MATLAB\RGBZ\1170.png'));
    elseif v==9
        im_in = double(imread('D:\image desioning\MATLAB\RGBZ\1180.png'));
    elseif v==10
        im_in = double(imread('D:\image desioning\MATLAB\RGBZ\1190.png'));
        %im_in =imread('D:\image desioning\MATLAB\RGBZ\10.png');
    elseif v==11
        im_in = double(imread('D:\image desioning\MATLAB\RGBZ\1200.png'));
    elseif v==12
        im_in = double(imread('D:\image desioning\MATLAB\RGBZ\1210.png'));
    elseif v==13
        im_in = double(imread('D:\image desioning\MATLAB\RGBZ\010.png'));
    elseif v==14
        im_in = double(imread('D:\image desioning\MATLAB\RGBZ\011.png'));
    elseif v==15
        im_in = double(imread('D:\image desioning\MATLAB\RGBZ\012.png'));   
    elseif v==16
        im_in = double(imread('D:\image desioning\MATLAB\RGBZ\013.png'));
    elseif v==17
        im_in = double(imread('D:\image desioning\MATLAB\RGBZ\014.png'));
    elseif v==18
        im_in = double(imread('D:\image desioning\MATLAB\RGBZ\015.png'));
    elseif v==19
        im_in = double(imread('D:\image desioning\MATLAB\NLGBT\full-half-third\midd1-f.png'));
    elseif v==20
        im_in = double(imread('D:\image desioning\MATLAB\NLGBT\full-half-third\midd2-f.png'));
    elseif v==21
        im_in = double(imread('D:\image desioning\MATLAB\NLGBT\full-half-third\moebius-f.png'));
    elseif v==22
        im_in = double(imread('D:\image desioning\MATLAB\NLGBT\full-half-third\monopoly-f.png'));
    elseif v==23
        im_in = double(imread('D:\image desioning\MATLAB\NLGBT\full-half-third\plastic-f.png'));
    elseif v==24
        im_in = double(imread('D:\image desioning\MATLAB\NLGBT\full-half-third\reindeer-f.png'));
    elseif v==25
        im_in = double(imread('D:\image desioning\MATLAB\NLGBT\full-half-third\rocks1-f.png'));   
    elseif v==26
        im_in = double(imread('D:\image desioning\MATLAB\NLGBT\full-half-third\rocks2-f.png'));
    elseif v==27
        im_in = double(imread('D:\image desioning\MATLAB\NLGBT\full-half-third\teddy-f.pgm'));
    elseif v==28
        im_in = double(imread('D:\image desioning\MATLAB\NLGBT\full-half-third\wood1-f.png'));
    elseif v==29
        im_in = double(imread('D:\image desioning\MATLAB\NLGBT\full-half-third\wood2-f.png'));
    end
    v

    sigma_list =[10,30,50];  % noise variance
    %im_in = double(imread(['cloth4.png']));  % the input original image
    %im_noisy = double(imread('images/cones1.png'));  % the input original image
    
    for j = 1:length(sigma_list)
        nSig = sigma_list(j)
        im_noisy = im_in + randn(size(im_in)) * nSig; % generate the noisy image
        Psnr  =  csnr( im_noisy, im_in, 0, 0 );
        %fprintf( 'Noisy Image: nSig = %2.3f, PSNR = %2.2f \n\n\n', nSig, Psnr );
        Par   = ParSet(nSig);    
        im_out = WNNM_DeNoising( im_noisy, im_in, Par ); 
%csnr( im_in, im_out, 0, 0 );
%ssim( im_in/256, im_out/256);
        PSNR(30*(v-1)+j) = csnr( im_in, im_out, 0, 0 );
        SSIM(30*(v-1)+j) = ssim( im_in/256, im_out/256);
        %imshow(im2uint8(im_out/256));
    end
end