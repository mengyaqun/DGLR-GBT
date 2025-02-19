clc;
clear;

PS=zeros(30);
SS=zeros(30);
Tim=zeros(30);
for i = 113:118
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
            case 31
                filename = '31';                 
            case 32
                filename = '32';
            case 33
                filename = '33';
            case 34
                filename = '34';    
            case 35
                filename = '35';  
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

filename    

randn ('seed',0);

Sigma_Num         =       [30,50]; 

Sigma             =       Sigma_Num(j)


 if  Sigma ==10
     
    gamma = 0.1; lambda = 0.1;  mu = 0.06;  c1 = 1.2;  c2 = 0.6;

 elseif  Sigma ==15


     gamma = 0.09; lambda = 0.04;  mu = 0.1; c1 = 1;  c2 = 1.2;

 elseif  Sigma ==20


     gamma = 0.09; lambda = 0.04;  mu = 0.1; c1 = 1;  c2 = 1.2;

 elseif Sigma ==25
     
          gamma = 0.09; lambda = 0.09;  mu = 0.009;  c1 = 1;  c2 = 0.1;
          
 elseif Sigma ==30
     
          gamma = 0.09; lambda = 0.09;  mu = 0.009;  c1 = 1;  c2 = 0.1;
            
 elseif Sigma ==40
     
          gamma = 0.08; lambda = 0.08;  mu = 0.006; c1 = 1;  c2 = 0.3;
             
 elseif  Sigma ==50
     

          gamma = 0.09; lambda = 0.08;  mu = 0.004;  c1 = 1;  c2 = 1.7;
                 
 elseif  Sigma == 75
     
           gamma = 0.1; lambda = 0.08;  mu = 0.002;   c1 = 1;  c2 = 1.7;
 else
     
          gamma = 0.08; lambda = 0.03;  mu = 0.007;   c1 = 1;  c2 = 0.4;
 end


 [filename, Sigma,   PSNR,FSIM,SSIM, Time_s, jj, Err_or]     =  LR_GSC_Denoising_Test (filename, Sigma, gamma, lambda, mu, c1, c2); 

  PS((i-100)*30+j) = PSNR;
  SS((i-100)*30+j) = SSIM;
 end
end






         