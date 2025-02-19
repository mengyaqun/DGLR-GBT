clc
clear
m_20=0; 
m_30=0;    
m_40=0;  
m_10=0;  
m_50=0;    
m_60=0;  
m_70=0;
m_80=0;  
m_90=0;  
m_100=0;    
m_110=0;  
m_120=0;


All_data_Results_2_20 = cell(1,200);
All_data_Results_2_30 = cell(1,200);
All_data_Results_2_40 = cell(1,200);
All_data_Results_2_10 = cell(1,200);

All_data_Results_2_50 = cell(1,200);
All_data_Results_2_60 = cell(1,200);
All_data_Results_2_70 = cell(1,200);

All_data_Results_2_80 = cell(1,200);
All_data_Results_2_90 = cell(1,200);

All_data_Results_2_100 = cell(1,200);
All_data_Results_2_110 = cell(1,200);
All_data_Results_2_120 = cell(1,200);



cur = cd;
addpath(genpath(cur));


for i = 1:100
    
ImageNum =i;

switch ImageNum
    
         case 1
                filename = '201';
            case 2
                filename = '202';
            case 3
                filename = '203';
            case 4
                filename = '204';    
            case 5
                filename = '205'; 
                
            case 6
                filename = '206';
            case 7
                filename = '207';
            case 8
                filename = '208';
            case 9
                filename = '209';    
            case 10
                filename = '210'; 
                
            case 11
                filename = '211';
            case 12
                filename = '212';
            case 13
                filename = '213';     
                
            case 14
                filename = '214';
            case 15
                filename = '215';
            case 16
                filename = '216';
            case 17
                filename = '217';    
            case 18
                filename = '218'; 
                
            case 19
                filename = '219';
            case 20
                filename = '220';
                
            case 21
                filename = '221';                     
            case 22
                filename = '222';
            case 23
                filename = '223';
            case 24
                filename = '224';
            case 25
                filename = '225';    
            case 26
                filename = '226';                
            case 27
                filename = '227';
            case 28
                filename = '228';
            case 29
                filename = '229';    
            case 30
                filename = '230';   
                
            case 31
                filename = '231';                     
            case 32
                filename = '232';
            case 33
                filename = '233';
            case 34
                filename = '234';
            case 35
                filename = '235';    
            case 36
                filename = '236';                
            case 37
                filename = '237';
            case 38
                filename = '238';
            case 39
                filename = '239';    
            case 40
                filename = '240';  
                
            case 41
                filename = '241';                     
            case 42
                filename = '242';
            case 43
                filename = '243';
            case 44
                filename = '244';
            case 45
                filename = '245';    
            case 46
                filename = '246';                
            case 47
                filename = '247';
            case 48
                filename = '248';
            case 49
                filename = '249';    
            case 50
                filename = '250';  
                          
            case 51
                filename = '251';                     
            case 52
                filename = '252';
            case 53
                filename = '253';
            case 54
                filename = '254';
            case 55
                filename = '255';    
            case 56
                filename = '256';                
            case 57
                filename = '257';
            case 58
                filename = '258';
            case 59
                filename = '259';    
            case 60
                filename = '260';   
                
            case 61
                filename = '261';                     
            case 62
                filename = '262';
            case 63
                filename = '263';
            case 64
                filename = '264';
            case 65
                filename = '265';    
            case 66
                filename = '266';                
            case 67
                filename = '267';
            case 68
                filename = '268';
            case 69
                filename = '269';    
            case 70
                filename = '270';        
                
            case 71
                filename = '271';                     
            case 72
                filename = '272';
            case 73
                filename = '273';
            case 74
                filename = '274';
            case 75
                filename = '275';    
            case 76
                filename = '276';                
            case 77
                filename = '277';
            case 78
                filename = '278';
            case 79
                filename = '279';    
            case 80
                filename = '280';                  
                  
                
             case 81
                filename = '281';                     
            case 82
                filename = '282';
            case 83
                filename = '283';
            case 84
                filename = '284';
            case 85
                filename = '285';    
            case 86
                filename = '286';                
            case 87
                filename = '287';
            case 88
                filename = '288';
            case 89
                filename = '289';    
            case 90
                filename = '290';       
                
            case 91
                filename = '291';                     
            case 92
                filename = '292';
            case 93
                filename = '293';
            case 94
                filename = '294';
            case 95
                filename = '295';    
            case 96
                filename = '296';                
            case 97
                filename = '297';
            case 98
                filename = '298';
            case 99
                filename = '299';    
            case 100
                filename = '300';   
          
end

for j  = 1:10
        
 

randn ('seed',0);

filename


JPEG_Quality_Num            = [ 10, 15, 20, 30, 40, 50, 60, 70, 80, 90];

      
JPEG_Quality            =      JPEG_Quality_Num (j)


  

if  j ==1  %10

               

 [filename, JPEG_Quality,PSNR_Final,FSIM_Final,SSIM_Final]     =  RRC_Deblock_Test (filename, JPEG_Quality ); 

 m_30= m_30+1;
 
 s=strcat('A',num2str(m_30));
 
 All_data_Results_2_30{m_30}=  {filename, JPEG_Quality, PSNR_Final,FSIM_Final,SSIM_Final};
 
 xlswrite('RRC_Deblock_Mq_10_BSD.xls', All_data_Results_2_30{m_30},'sheet1',s);
 
 
 elseif  j ==2  %15


 [filename, JPEG_Quality,PSNR_Final,FSIM_Final,SSIM_Final]     =  RRC_Deblock_Test (filename, JPEG_Quality ); 

 m_40= m_40+1;
 
 s=strcat('A',num2str(m_40));
 
 All_data_Results_2_40{m_40}   =  {filename, JPEG_Quality, PSNR_Final,FSIM_Final,SSIM_Final};
 
 xlswrite('RRC_Deblock_Mq_15_BSD.xls', All_data_Results_2_40{m_40},'sheet1',s);
 
 
 
  elseif  j ==3  % 20


 [filename, JPEG_Quality,PSNR_Final,FSIM_Final,SSIM_Final]     =  RRC_Deblock_Test (filename, JPEG_Quality ); 

 m_50= m_50+1;
 
 s=strcat('A',num2str(m_50));
 
 All_data_Results_2_50{m_50}   = {filename, JPEG_Quality, PSNR_Final,FSIM_Final,SSIM_Final};
 
 xlswrite('RRC_Deblock_Mq_20_BSD.xls', All_data_Results_2_50{m_50},'sheet1',s);
 
 
 
   elseif  j ==4  % 30
     

 [filename, JPEG_Quality,PSNR_Final,FSIM_Final,SSIM_Final]     =  RRC_Deblock_Test (filename, JPEG_Quality ); 

 m_60= m_60+1;
 
 s=strcat('A',num2str(m_60));
 
 All_data_Results_2_60{m_60}   = {filename, JPEG_Quality, PSNR_Final,FSIM_Final,SSIM_Final};
 
 xlswrite('RRC_Deblock_Mq_30_BSD.xls', All_data_Results_2_60{m_60},'sheet1',s);    
     
 
 
 elseif  j ==5 % 40
              
 [filename, JPEG_Quality,PSNR_Final,FSIM_Final,SSIM_Final]     =  RRC_Deblock_Test (filename, JPEG_Quality ); 

 m_70= m_70+1;
 
 s=strcat('A',num2str(m_70));
 
 All_data_Results_2_70{m_70}   =  {filename, JPEG_Quality, PSNR_Final,FSIM_Final,SSIM_Final};
 
 xlswrite('RRC_Deblock_Mq_40_BSD.xls', All_data_Results_2_70{m_70},'sheet1',s);    
 
 
  elseif  j ==6 % 50
              
 [filename, JPEG_Quality,PSNR_Final,FSIM_Final,SSIM_Final]     =  RRC_Deblock_Test (filename, JPEG_Quality ); 

 m_80= m_80+1;
 
 s=strcat('A',num2str(m_80));
 
 All_data_Results_2_80{m_80}   =  {filename, JPEG_Quality, PSNR_Final,FSIM_Final,SSIM_Final};
 
 xlswrite('RRC_Deblock_Mq_50_BSD.xls', All_data_Results_2_80{m_80},'sheet1',s);  
 
 
 
  elseif  j ==7 % 60
              
 [filename, JPEG_Quality,PSNR_Final,FSIM_Final,SSIM_Final]     =  RRC_Deblock_Test (filename, JPEG_Quality ); 

 m_90= m_90+1;
 
 s=strcat('A',num2str(m_90));
 
 All_data_Results_2_90{m_90}   =  {filename, JPEG_Quality, PSNR_Final,FSIM_Final,SSIM_Final};
 
 xlswrite('RRC_Deblock_Mq_60_BSD.xls', All_data_Results_2_90{m_90},'sheet1',s);  
 
 
 
  elseif  j ==8  % 70
              
 [filename, JPEG_Quality,PSNR_Final,FSIM_Final,SSIM_Final]     =  RRC_Deblock_Test (filename, JPEG_Quality ); 

 m_100= m_100+1;
 
 s=strcat('A',num2str(m_100));
 
 All_data_Results_2_100{m_100}   =  {filename, JPEG_Quality, PSNR_Final,FSIM_Final,SSIM_Final};
 
 xlswrite('RRC_Deblock_Mq_70_BSD.xls', All_data_Results_2_100{m_100},'sheet1',s);   
 
 
 
  elseif  j ==9 % 80
              
 [filename, JPEG_Quality,PSNR_Final,FSIM_Final,SSIM_Final]     =  RRC_Deblock_Test (filename, JPEG_Quality ); 

 m_110= m_110+1;
 
 s=strcat('A',num2str(m_110));
 
 All_data_Results_2_110{m_110}   =  {filename, JPEG_Quality, PSNR_Final,FSIM_Final,SSIM_Final};
 
 xlswrite('RRC_Deblock_Mq_80_BSD.xls', All_data_Results_2_110{m_110},'sheet1',s);  
 
 
 else
     
     
  [filename, JPEG_Quality,PSNR_Final,FSIM_Final,SSIM_Final]     =  RRC_Deblock_Test (filename, JPEG_Quality ); 

 m_120= m_120+1;
 
 s=strcat('A',num2str(m_120));
 
 All_data_Results_2_120{m_120}   =  {filename, JPEG_Quality, PSNR_Final,FSIM_Final,SSIM_Final};
 
 xlswrite('RRC_Deblock_Mq_90_BSD.xls', All_data_Results_2_120{m_120},'sheet1',s);      
 
 
          
     
 
 end
 

 clearvars -except filename i  m_20 All_data_Results_2_20 m_30 All_data_Results_2_30 m_40 All_data_Results_2_40...
     All_data_Results_2_10 m_10 All_data_Results_2_50 m_50 All_data_Results_2_60 m_60 All_data_Results_2_70 m_70...
     All_data_Results_2_80 m_80 All_data_Results_2_90 m_90 All_data_Results_2_100 m_100 All_data_Results_2_110 m_110 All_data_Results_2_120 m_120
end


 clearvars -except filename  m n m_20 All_data_Results_2_20 m_30 All_data_Results_2_30 m_40 All_data_Results_2_40...
     All_data_Results_2_10 m_10 All_data_Results_2_50 m_50 All_data_Results_2_60 m_60 All_data_Results_2_70 m_70...
     All_data_Results_2_80 m_80 All_data_Results_2_90 m_90 All_data_Results_2_100 m_100 All_data_Results_2_110 m_110 All_data_Results_2_120 m_120
end





         