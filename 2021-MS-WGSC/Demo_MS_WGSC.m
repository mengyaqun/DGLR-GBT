%--------------------------------------------------------------------------
% Implementation of the MS_WGSC algorithm for image denoising.
% Author:  Yang Ou, ouyang@my.swjtu.edu.cn
%          Southwest Jiaotong University
%--------------------------------------------------------------------------

clc;clear;close all;
% add the user toolboxes
%toolPath = pwd;
% Add functions
%addpath(toolPath, 'Utilities');

PS=zeros(30);
SS=zeros(30);

for v=1:12
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
        im_in = double(imread('D:\image desioning\MATLAB\RGBZ\010.png'));
    elseif v==8
        im_in = double(imread('D:\image desioning\MATLAB\RGBZ\011.png'));
    elseif v==9
        im_in = double(imread('D:\image desioning\MATLAB\RGBZ\012.png'));   
    elseif v==10
        im_in = double(imread('D:\image desioning\MATLAB\RGBZ\013.png'));
    elseif v==11
        im_in = double(imread('D:\image desioning\MATLAB\RGBZ\014.png'));
    elseif v==12
        im_in = double(imread('D:\image desioning\MATLAB\RGBZ\015.png'));    
    elseif v==13
        im_in = double(imread('D:\image desioning\MATLAB\RGBZ\1160.png'));
    elseif v==14
        im_in = double(imread('D:\image desioning\MATLAB\RGBZ\1170.png'));
    elseif v==15
        im_in = double(imread('D:\image desioning\MATLAB\RGBZ\1180.png'));
    elseif v==16
        im_in = double(imread('D:\image desioning\MATLAB\RGBZ\1190.png'));
        %im_in =imread('D:\image desioning\MATLAB\RGBZ\10.png');
    elseif v==17
        im_in = double(imread('D:\image desioning\MATLAB\RGBZ\1200.png'));
    elseif v==18
        im_in = double(imread('D:\image desioning\MATLAB\RGBZ\1210.png'));

    elseif v==19
        im_in = double(imread('D:\image desioning\MATLAB\2021-MS-WGSC\19.png'));
    elseif v==20
        im_in = double(imread('D:\image desioning\MATLAB\2021-MS-WGSC\20.png'));
    elseif v==21
        im_in = double(imread('D:\image desioning\MATLAB\2021-MS-WGSC\21.png'));
    elseif v==22
        im_in = double(imread('D:\image desioning\MATLAB\2021-MS-WGSC\22.png'));
    elseif v==23
        im_in = double(imread('D:\image desioning\MATLAB\2021-MS-WGSC\23.png'));
    elseif v==24
        im_in = double(imread('D:\image desioning\MATLAB\2021-MS-WGSC\24.png'));
    elseif v==25
        im_in = double(imread('D:\image desioning\MATLAB\2021-MS-WGSC\25.png'));
    elseif v==26
        im_in = double(imread('D:\image desioning\MATLAB\2021-MS-WGSC\26.png'));
    elseif v==27
        im_in = double(imread('D:\image desioning\MATLAB\2021-MS-WGSC\27.png'));
    elseif v==28
        im_in = double(imread('D:\image desioning\MATLAB\2021-MS-WGSC\28.png'));
    elseif v==29
        im_in = double(imread('D:\image desioning\MATLAB\2021-MS-WGSC\29.png'));
    elseif v==30
        im_in = double(imread('D:\image desioning\MATLAB\2021-MS-WGSC\30.png'));
    else
        break;
    end
    v
    for j  =  1:3
            
        Sigma_Num            = [10,30,50];
        
        Sigma            =      Sigma_Num (j);
        
        I0 = im_in; % Input image
        nSig = Sigma;                      % Noise level
        randn('seed', 0);
        Ynoi = I0 + randn(size(I0)) * nSig;
        PSNR    =  csnr( Ynoi, I0, 0, 0 );
        SSIM    =  cal_ssim( Ynoi, I0, 0, 0 );
         % Display the PSNR/SIMM of input images
        fprintf( 'Noisy Image: SD = %2.2f, PSNR = %2.2f, SSIM = %2.4f \n', nSig,PSNR,SSIM);
        
        % Parameter setting
        par = setParameters(nSig);
        [im_out,par] = MS_WGSC(I0, Ynoi, par);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        im_out(im_out>255)=255;
        im_out(im_out<0)=0;
        % calculate the PSNR
        
        PS((v-1)*30+j)  =  csnr( im_out, I0, 0, 0 );
        SS((v-1)*30+j)  =  cal_ssim( im_out, I0, 0, 0 );
        %csnr( im_out, I0, 0, 0 )
        %cal_ssim( im_out, I0, 0, 0 )
        %imshow(im_out/256)
      end
end
