function [filename, Sigma, PSNR_Final,SSIM_Final]     =  RRC_Test (filename, Sigma)

randn ('seed',0);


fn               =     [filename, '.png'];

I                =     imread(fn);

[~, ~, kk]       =     size (I);

if kk==3
    
    I     = rgb2gray (I);
       
end


Opts              =    Opts_Set (Sigma, I); % parameter setting

Opts.nim          =    Opts.I + Opts.nSig* randn(size( Opts.I ));

disp(sprintf('PSNR of the noisy image = %f \n', csnr(Opts.nim, Opts.I, 0, 0) ));


[Denoising , iter]              =    RRC_Denoising( Opts);

im  = Denoising{iter-1};


PSNR_Final       =   csnr (im, Opts.I,0,0);

SSIM_Final       =   cal_ssim (im, Opts.I,0,0);

%imshow(im/256)

if Sigma==10

Final_denoisng= strcat(filename,'_RRC','_sigma_',num2str(Sigma),'_PSNR_',num2str(PSNR_Final),'_SSIM_',num2str(SSIM_Final),'.png');

imwrite(uint8(im),strcat('./10_Result/',Final_denoisng));


elseif Sigma==20

Final_denoisng= strcat(filename,'_RRC','_sigma_',num2str(Sigma),'_PSNR_',num2str(PSNR_Final),'_SSIM_',num2str(SSIM_Final),'.png');

imwrite(uint8(im),strcat('./20_Result/',Final_denoisng));

elseif Sigma==30
    
Final_denoisng= strcat(filename,'_RRC','_sigma_',num2str(Sigma),'_PSNR_',num2str(PSNR_Final),'_SSIM_',num2str(SSIM_Final),'.png');

imwrite(uint8(im),strcat('./30_Result/',Final_denoisng));

elseif Sigma==40
    
Final_denoisng= strcat(filename,'_RRC','_sigma_',num2str(Sigma),'_PSNR_',num2str(PSNR_Final),'_SSIM_',num2str(SSIM_Final),'.png');

imwrite(uint8(im),strcat('./40_Result/',Final_denoisng));

elseif Sigma==50
    
Final_denoisng= strcat(filename,'_RRC','_sigma_',num2str(Sigma),'_PSNR_',num2str(PSNR_Final),'_SSIM_',num2str(SSIM_Final),'.png');

imwrite(uint8(im),strcat('./50_Result/',Final_denoisng));

elseif Sigma==75
    
Final_denoisng= strcat(filename,'_RRC','_sigma_',num2str(Sigma),'_PSNR_',num2str(PSNR_Final),'_SSIM_',num2str(SSIM_Final),'.png');

imwrite(uint8(im),strcat('./75_Result/',Final_denoisng));

else
    
Final_denoisng= strcat(filename,'_RRC','_sigma_',num2str(Sigma),'_PSNR_',num2str(PSNR_Final),'_SSIM_',num2str(SSIM_Final),'.png');

imwrite(uint8(im),strcat('./100_Result/',Final_denoisng));
end


end

