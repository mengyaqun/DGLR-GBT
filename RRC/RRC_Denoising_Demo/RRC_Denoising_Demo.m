clc;
clear;
PS=zeros(30);
SS=zeros(30);


for i = 113:118
i    
ImageNum =i;

switch ImageNum

            case 1
                filename = '1';
            case 2
                filename = '2';
            case 3
                filename = '3';
            case 4
                filename = '4';    
            case 5
                filename = '5';               
            case 6
                filename = '6';
            case 7
                filename = '7';
            case 8
                filename = '8';
            case 9
                filename = '9';    
            case 10
                filename = '10';                
            case 11
                filename = '11';
            case 12
                filename = '12';
            case 13
                filename = '13';                   
            case 14
                filename = '14';
            case 15
                filename = '15';
            case 16
                filename = '16';
            case 17
                filename = '17';    
            case 18
                filename = '18';                
            case 19
                filename = '19';
            case 20
                filename = '20';
            case 21
                filename = '21';     
            case 22
                filename = '22';
            case 23
                filename = '23';
            case 24
                filename = '24';
            case 25
                filename = '25';    
            case 26
                filename = '26';                 
            case 27
                filename = '27';
            case 28
                filename = '28';
            case 29
                filename = '29';    
            case 30
                filename = '30';
            case 113
                filename = '113';
            case 114
                filename = '114';    
            case 115
                filename = '115';               
            case 116
                filename = '116';
            case 117
                filename = '117';
            case 118
                filename = '118';
   
end

for j  =  1:2
    
randn ('seed',0);

Sigma_Num            = [30,50];

filename

Sigma            =      Sigma_Num (j)
    
 if  Sigma ==10
     
     
 [filename, Sigma, PSNR_Final,SSIM_Final]     =  RRC_Test (filename, Sigma); 
 
 %m_10= m_10+1;
 
 elseif  Sigma ==15
     
 [filename, Sigma, PSNR_Final,SSIM_Final]     =  RRC_Test (filename, Sigma); 
 
% m_15= m_15+1;
  
 elseif  Sigma ==20
     
 [filename, Sigma, PSNR_Final,SSIM_Final]     =  RRC_Test (filename, Sigma); 
 
% m_20= m_20+1;

 elseif  Sigma ==25
     
 [filename, Sigma, PSNR_Final,SSIM_Final]     =  RRC_Test (filename, Sigma); 
 
% m_25= m_25+1;
 
 elseif  Sigma ==30
     
 [filename, Sigma, PSNR_Final,SSIM_Final]     =  RRC_Test (filename, Sigma); 
 
% m_30= m_30+1;
 
 elseif  Sigma ==40
    
 [filename, Sigma, PSNR_Final,SSIM_Final]     =  RRC_Test (filename, Sigma); 
 
% m_40= m_40+1;

 elseif  Sigma ==50
   
 [filename, Sigma, PSNR_Final,SSIM_Final]     =  RRC_Test (filename, Sigma); 
 
 %m_50= m_50+1;
 
 elseif  Sigma ==75
     
 [filename, Sigma, PSNR_Final,SSIM_Final]     =  RRC_Test (filename, Sigma); 
 
% m_60= m_60+1;
 
 else

 [filename, Sigma, PSNR_Final,SSIM_Final]     =  RRC_Test (filename, Sigma); 
 
% m_70= m_70+1;
 

 end
 PSNR_Final;
 SSIM_Final;
	PS((i-100)*30+j) = PSNR_Final;
	SS((i-100)*30+j) = SSIM_Final;
end

end





         