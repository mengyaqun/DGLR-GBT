clc;clear;
PS=zeros(30);
SS=zeros(30);
for i = 2 : 2  
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
end

  for j   = 1:1    
	randn ('seed',0);
	filename
	Sigma_Num            =       [10];
	Sigma                =       Sigma_Num (j)
	c1 = 1;
	c2 = 1;
	
	
 	if  Sigma <=10      
      	     gamma  = 0.1;      lambda  =  0.5;   mu  =  0.3;   
	elseif  Sigma <=20          
             gamma  = 0.1;      lambda  =  0.4;   mu  =  0.4;   
 	elseif  Sigma <=30   
             gamma  = 0.1;      lambda  =  0.4;   mu  =  0.4;    
 	elseif Sigma <=40    
             gamma  = 0.1;      lambda  =  0.3;   mu  =  0.7;   
 	elseif  Sigma <=50
             gamma  = 0.1;      lambda  =  0.3;   mu  =  0.7;   
 	elseif   Sigma <= 75
             gamma  = 0.1;      lambda  =  0.3;   mu  =  0.8;   
	else        
             gamma  = 0.1;      lambda  =  0.3;   mu  =  0.7;    
    end

  [filename, Sigma,   PSNR,FSIM,SSIM, Time_s, jj]     =  LGSR_Denoising_Test (filename, Sigma, gamma, lambda, mu, c1, c2);
  PS((i-1)*30+j) = PSNR;
  SS((i-1)*30+j) = SSIM;
  PSNR
  SSIM
  
  end

end