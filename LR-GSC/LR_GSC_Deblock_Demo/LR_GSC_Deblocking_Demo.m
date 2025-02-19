clc
clear
m_10_1=0; 
m_10_2=0; 
m_10_3=0; 
m_20_1=0; 
m_20_2=0; 
m_20_3=0; 
m_30_1=0; 
m_30_2=0; 
m_30_3=0; 
m_40_1=0;
m_40_2=0; 
m_40_3=0; 
m_50_1=0;
m_50_2=0; 
m_50_3=0; 
m_75_1=0;
m_75_2=0; 
m_75_3=0; 
m_100_1=0;
m_100_2=0; 
m_100_3=0; 

All_data_Results_10_1 = cell(1,200);
All_data_Results_10_2 = cell(1,200);
All_data_Results_10_3 = cell(1,200);

All_data_Results_20_1 = cell(1,200);
All_data_Results_20_2 = cell(1,200);
All_data_Results_20_3 = cell(1,200);

All_data_Results_30_1 = cell(1,200);
All_data_Results_30_2 = cell(1,200);
All_data_Results_30_3 = cell(1,200);

All_data_Results_40_1 = cell(1,200);
All_data_Results_40_2 = cell(1,200);
All_data_Results_40_3 = cell(1,200);

All_data_Results_50_1 = cell(1,200);
All_data_Results_50_2 = cell(1,200);
All_data_Results_50_3 = cell(1,200);

All_data_Results_75_1 = cell(1,200);
All_data_Results_75_2 = cell(1,200);
All_data_Results_75_3 = cell(1,200);

All_data_Results_100_1 = cell(1,200);
All_data_Results_100_2 = cell(1,200);
All_data_Results_100_3 = cell(1,200);



for i = 1:3
ImageNum =i;

switch ImageNum
         case 1
                filename = 'Barbara256';
            case 2
                filename = 'House256';
            case 3
                filename = 'Leaves256';

end
for j  =  1:10
    
  
 
        filename
        
    
randn ('seed',0);

JPEG_Quality_Num            = [ 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90];



JPEG_Quality            =      JPEG_Quality_Num (j)



             


if JPEG_Quality<=1
    
    
   lambda  = 0.5;  alpha = 0.0007;  beta = 1;  mu = 1.6; c1 = 0.9; c2 = 1.7;   QF = 0.3 ;
    
elseif JPEG_Quality<=5
    
  lambda  = 0.5;  alpha = 0.002;  beta = 1.1;  mu = 1.4; c1 = 1; c2 = 1.4;     QF = 0.3 ;

elseif JPEG_Quality<=10
    
  lambda  = 0.6;  alpha = 0.002;  beta = 0.8;  mu = 0.9;  c1 = 1; c2 = 1.9;     QF = 0.25 ;
    
elseif JPEG_Quality<=20
    
  lambda  = 0.7;  alpha = 0.002;  beta = 1;  mu = 0.8;    c1 = 0.7; c2 = 1.4;   QF = 0.25 ;
    
elseif JPEG_Quality<=30
    
  lambda  = 0.8;  alpha = 0.002;  beta = 1;  mu = 0.8;    c1 = 0.9; c2 = 2;     QF = 0.25 ;
    
elseif JPEG_Quality<=40
    
 lambda  = 0.8;  alpha = 0.002;  beta = 0.9;  mu = 0.8;    c1 = 1.5; c2 = 1.7;   QF = 0.25 ;
    
else
 lambda  = 0.8;  alpha = 0.002;  beta = 0.9;  mu = 0.8;      c1 = 1.5; c2 = 1.7;  QF = 0.25 ;
 
end




 if  JPEG_Quality== 1
     
     

  [filename, JPEG_Quality,lambda, alpha, beta, mu, c1, c2,JPEG_PSNR, PSNR_Final,FSIM_Final,SSIM_Final, iter, diff, Time_s]     =  LR_GSC_Deblock_Test (filename, JPEG_Quality, lambda, alpha, beta, mu,  c1, c2, QF);
 
  m_10_1= m_10_1+1;
 
 s=strcat('A',num2str(m_10_1));
 
 All_data_Results_10_1{m_10_1}= {filename, JPEG_Quality,  PSNR_Final, FSIM_Final,SSIM_Final, iter, diff, Time_s};
 
 xlswrite('LR_GSC_JPEG_Quality_1_test_Final.xls', All_data_Results_10_1{m_10_1},'sheet1',s);
 
 

 
 elseif  JPEG_Quality==5
     
     

  [filename, JPEG_Quality,lambda, alpha, beta, mu, c1, c2,JPEG_PSNR, PSNR_Final,FSIM_Final,SSIM_Final, iter, diff, Time_s]     =  LR_GSC_Deblock_Test (filename, JPEG_Quality, lambda, alpha, beta, mu,  c1, c2, QF);
 
 m_20_1= m_20_1+1;
 
 s=strcat('A',num2str(m_20_1));
 
 All_data_Results_20_1{m_20_1}= {filename, JPEG_Quality,  PSNR_Final, FSIM_Final,SSIM_Final, iter, diff, Time_s};
 
 xlswrite('LR_GSC_JPEG_Quality_5_test_Final.xls', All_data_Results_20_1{m_20_1},'sheet1',s);
 
 
 
  elseif  JPEG_Quality==10
      

  [filename, JPEG_Quality,lambda, alpha, beta, mu, c1, c2,JPEG_PSNR, PSNR_Final,FSIM_Final,SSIM_Final, iter, diff, Time_s]     =  LR_GSC_Deblock_Test (filename, JPEG_Quality, lambda, alpha, beta, mu,  c1, c2, QF);
 
 m_30_1= m_30_1+1;
 
 s=strcat('A',num2str(m_30_1));
 
 All_data_Results_30_1{m_30_1}= {filename, JPEG_Quality,  PSNR_Final, FSIM_Final,SSIM_Final, iter, diff, Time_s};
 
 xlswrite('LR_GSC_JPEG_Quality_10_test_Final.xls', All_data_Results_30_1{m_30_1},'sheet1',s);

   elseif  JPEG_Quality==20
       

   
  [filename, JPEG_Quality,lambda, alpha, beta, mu, c1, c2,JPEG_PSNR, PSNR_Final,FSIM_Final,SSIM_Final, iter, diff, Time_s]     =  LR_GSC_Deblock_Test (filename, JPEG_Quality, lambda, alpha, beta, mu,  c1, c2, QF);
 
 m_40_1= m_40_1+1;
 
 s=strcat('A',num2str(m_40_1));
 
 All_data_Results_40_1{m_40_1}= {filename, JPEG_Quality,  PSNR_Final, FSIM_Final,SSIM_Final, iter, diff, Time_s};
 
 xlswrite('LR_GSC_JPEG_Quality_20_test_Final.xls', All_data_Results_40_1{m_40_1},'sheet1',s);
 
 
   
 elseif  JPEG_Quality==30
     

  [filename, JPEG_Quality,lambda, alpha, beta, mu, c1, c2,JPEG_PSNR, PSNR_Final,FSIM_Final,SSIM_Final, iter, diff, Time_s]     =  LR_GSC_Deblock_Test (filename, JPEG_Quality, lambda, alpha, beta, mu,  c1, c2, QF);
 
 m_50_1= m_50_1+1;
 
 s=strcat('A',num2str(m_50_1));
 
 All_data_Results_50_1{m_50_1}= {filename, JPEG_Quality,  PSNR_Final, FSIM_Final,SSIM_Final, iter, diff, Time_s};
 
 xlswrite('LR_GSC_JPEG_Quality_30_test_Final.xls', All_data_Results_50_1{m_50_1},'sheet1',s);
 
 
 
 
 
 elseif  JPEG_Quality==40
     
     

  [filename, JPEG_Quality,lambda, alpha, beta, mu, c1, c2,JPEG_PSNR, PSNR_Final,FSIM_Final,SSIM_Final, iter, diff, Time_s]     =  LR_GSC_Deblock_Test (filename, JPEG_Quality, lambda, alpha, beta, mu,  c1, c2, QF);
 
 m_75_1= m_75_1+1;
 
 s=strcat('A',num2str(m_75_1));
 
 All_data_Results_75_1{m_75_1}= {filename, JPEG_Quality,  PSNR_Final, FSIM_Final,SSIM_Final, iter, diff, Time_s};
 
 xlswrite('LR_GSC_JPEG_Quality_40_1_test_Final.xls', All_data_Results_75_1{m_75_1},'sheet1',s);
 
 
 elseif JPEG_Quality==50
         
     
  [filename, JPEG_Quality,lambda, alpha, beta, mu, c1, c2,JPEG_PSNR, PSNR_Final,FSIM_Final,SSIM_Final, iter, diff, Time_s]     =  LR_GSC_Deblock_Test (filename, JPEG_Quality, lambda, alpha, beta, mu,  c1, c2, QF);
 
 m_75_2= m_75_2+1;
 
 s=strcat('A',num2str(m_75_2));
 
 All_data_Results_75_2{m_75_2}= {filename, JPEG_Quality,  PSNR_Final, FSIM_Final,SSIM_Final, iter, diff, Time_s};
 
 xlswrite('LR_GSC_JPEG_Quality_50_test_Final.xls', All_data_Results_75_2{m_75_2},'sheet1',s);         
          
 elseif JPEG_Quality==60
     
  [filename, JPEG_Quality,lambda, alpha, beta, mu, c1, c2,JPEG_PSNR, PSNR_Final,FSIM_Final,SSIM_Final, iter, diff, Time_s]     =  LR_GSC_Deblock_Test (filename, JPEG_Quality, lambda, alpha, beta, mu,  c1, c2, QF);
 
 m_75_3= m_75_3+1;
 
 s=strcat('A',num2str(m_75_3));
 
 All_data_Results_75_3{m_75_3}= {filename, JPEG_Quality,  PSNR_Final, FSIM_Final,SSIM_Final, iter, diff, Time_s};
 
 xlswrite('LR_GSC_JPEG_Quality_60_test_Final.xls', All_data_Results_75_3{m_75_3},'sheet1',s);        
 
 
         
 elseif JPEG_Quality==70

     
  [filename, JPEG_Quality,lambda, alpha, beta, mu, c1, c2,JPEG_PSNR, PSNR_Final,FSIM_Final,SSIM_Final, iter, diff, Time_s]     =  LR_GSC_Deblock_Test (filename, JPEG_Quality, lambda, alpha, beta, mu,  c1, c2, QF);
 
 m_100_1= m_100_1+1;
 
 s=strcat('A',num2str(m_100_1));
 
 All_data_Results_100_1{m_100_1}= {filename, JPEG_Quality,  PSNR_Final, FSIM_Final,SSIM_Final, iter, diff, Time_s};
 
 xlswrite('LR_GSC_JPEG_Quality_70_test_Final.xls', All_data_Results_100_1{m_100_1},'sheet1',s);
 
 
 
 
 elseif JPEG_Quality==80

     
  [filename, JPEG_Quality,lambda, alpha, beta, mu, c1, c2,JPEG_PSNR, PSNR_Final,FSIM_Final,SSIM_Final, iter, diff, Time_s]     =  LR_GSC_Deblock_Test (filename, JPEG_Quality, lambda, alpha, beta, mu,  c1, c2, QF);
 
 m_100_2= m_100_2+1;
 
 s=strcat('A',num2str(m_100_2));
 
 All_data_Results_100_2{m_100_2}= {filename, JPEG_Quality,  PSNR_Final, FSIM_Final,SSIM_Final, iter, diff, Time_s};
 
 xlswrite('LR_GSC_JPEG_Quality_80_test_Final.xls', All_data_Results_100_2{m_100_2},'sheet1',s);         
 
 
 
          
     else

     
  [filename, JPEG_Quality,lambda, alpha, beta, mu, c1, c2,JPEG_PSNR, PSNR_Final,FSIM_Final,SSIM_Final, iter, diff, Time_s]     =  LR_GSC_Deblock_Test (filename, JPEG_Quality, lambda, alpha, beta, mu,  c1, c2, QF);
 
 m_100_3= m_100_3+1;
 
 s=strcat('A',num2str(m_100_3));
 
 All_data_Results_100_3{m_100_3}= {filename, JPEG_Quality,  PSNR_Final, FSIM_Final,SSIM_Final, iter, diff, Time_s};
 
 xlswrite('LR_GSC_JPEG_Quality_90_test_Final.xls', All_data_Results_100_3{m_100_3},'sheet1',s);                
         


 end
 

clearvars -except filename i m_10_1 m_10_2 m_10_3 m_20_1 m_20_2 m_20_3 m_30_1 m_30_2 m_30_3 m_40_1 m_40_2 m_40_3...
m_50_1 m_50_2 m_50_3 m_75_1 m_75_2 m_75_3 m_100_1 m_100_2 m_100_3 All_data_Results_10_1...
All_data_Results_10_2 All_data_Results_10_3 All_data_Results_20_1 All_data_Results_20_2...
All_data_Results_20_3 All_data_Results_30_1 All_data_Results_30_2 All_data_Results_30_3...
All_data_Results_40_1 All_data_Results_40_2 All_data_Results_40_3 All_data_Results_50_1...
All_data_Results_50_2 All_data_Results_50_3 All_data_Results_75_1 All_data_Results_75_2...
All_data_Results_75_3 All_data_Results_100_1 All_data_Results_100_2 All_data_Results_100_3
end


clearvars -except filename m_10_1 m_10_2 m_10_3 m_20_1 m_20_2 m_20_3 m_30_1 m_30_2 m_30_3 m_40_1 m_40_2 m_40_3...
m_50_1 m_50_2 m_50_3 m_75_1 m_75_2 m_75_3 m_100_1 m_100_2 m_100_3 All_data_Results_10_1...
All_data_Results_10_2 All_data_Results_10_3 All_data_Results_20_1 All_data_Results_20_2...
All_data_Results_20_3 All_data_Results_30_1 All_data_Results_30_2 All_data_Results_30_3...
All_data_Results_40_1 All_data_Results_40_2 All_data_Results_40_3 All_data_Results_50_1...
All_data_Results_50_2 All_data_Results_50_3 All_data_Results_75_1 All_data_Results_75_2...
All_data_Results_75_3 All_data_Results_100_1 All_data_Results_100_2 All_data_Results_100_3
end






         