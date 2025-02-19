%{
clear;clc;
 
%pauseTime = 0.01;

data_path = "D:\image desioning\MATLAB\RGBZ";
ext = ["*.jpg", "*.png", "*.jpeg"];
filePaths   =  [];
for i = 1 : length(ext)
    filePaths = cat(1,filePaths, dir(fullfile(data_path,ext(i))));
end

noise_leval = [10,15,20,25,30,40,50];
length(filePaths)
for i = 1:length(noise_leval)
    PSNRs = [];
    SSIMs = [];
    sigma = noise_leval(i)
    for j = 1:length(filePaths)
        y = imread(filePaths(j).name);
        if length(size(y)) > 2
            y = rgb2gray(y);
        end
        y = im2double(y);
        z = y + (sigma/255)*randn(size(y));
        %         ͼ  
        
        [PSNR,SSIM,y_est] = BM3D(y, z, sigma, 'np', 0);
        PSNRs(j) = PSNR;
        SSIMs(j) = SSIM;
       % imshow(cat(2,im2uint8(y),im2uint8(z),im2uint8(y_est)));
        %title([num2str(sigma),'   ', filePaths(j).name,'    ',num2str(PSNR,'%2.2f'),'dB','    ',num2str(SSIMs(j),'%2.4f')])
        %drawnow;
        %pause(pauseTime)
    end
    disp(["sigma:", sigma, " psnr:", mean(PSNRs), "  ssim:", mean(SSIMs)]);
end

%}

clc;
clear;
PSNR=zeros(30);
SSIM=zeros(30);

for v=13:18
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
        im_in = double(imread('D:\image desioning\MATLAB\RGBZ\19.jpg'));
    elseif v==20
        im_in = double(imread('D:\image desioning\MATLAB\RGBZ\20.jpg'));
    elseif v==21
        im_in = double(imread('D:\image desioning\MATLAB\RGBZ\21.jpg'));
    elseif v==22
        im_in = double(imread('D:\image desioning\MATLAB\RGBZ\22.png'));
    elseif v==23
        im_in = double(imread('D:\image desioning\MATLAB\RGBZ\23.png'));
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
    sigma_list =[10,15,20,25,30,40,50];  % noise variance
   % im_in = double(imread('cones2.png'));  % the input original image
    size(im_in)
    %im_noisy = double(imread('images/cones1.png'));  % the input original image
    for j = 1:length(sigma_list)
        nSig = sigma_list(j);
        
        if length(size(im_in)) > 2
            im_in = rgb2gray(im_in);
            im_in;
        end
        %imshow(im2uint8(im_in/256))
        im_in = im2double(im_in);%*256;
        im_noisy = im_in + randn(size(im_in)) * nSig; % generate the noisy image
        %psnr(im_noisy,im_in)
        %ssim(im_noisy,im_in)
        [PS,SS,im_out] = BM3D(im_in, im_noisy, nSig, 'np', 0);
        PSNR(30*(v-1)+j) = PS;
        SSIM(30*(v-1)+j) = SS;
        %PS;
        %SS;
        %imshow(im2uint8(im_noisy/256));
        %axes('Position',[0.4 0.3 0.3 0.25]);
    end
end

