function  [filename, JPEG_Quality,  PSNR_Final, PSNR_B, FSIM_Final,SSIM_Final]     =  RRC_Deblock_Test (filename, JPEG_Quality)
 
           randn ('seed',0);              
                 
                fn               =     [filename, '.jpg'];

                I                =     imread(fn);
                
               [ww, hh, kk]        =     size (I);
               
               I                   =    imresize (I, [ww-1, hh-1]);                
                
                colorI           =     I;

              if kk==3
    
                     I           =      rgb2gray (I);
                     
                     y_yuv       =      rgb2ycbcr(colorI);                     

                x_2_yuv(:,:,2)   =       y_yuv(:,:,2); % Copy U Componet
                  
                x_2_yuv(:,:,3)   =       y_yuv(:,:,3); % Copy V Componet                   
              end 
 
              
              
                Par                     =    Par_Set (I, JPEG_Quality);   

 %-------------------------------------JPEG Operator------------------------------------------%     
                           jpeg_name  = [filename '_jeg_q' num2str(JPEG_Quality)];
                           
                           imwrite(I, [jpeg_name '.jpg'], 'Quality',  JPEG_Quality);
                           
                           JPEG_info = imfinfo([jpeg_name '.jpg']);
 
                           JPEG_image = imread([jpeg_name '.jpg']);
                          
              Par.nim                    =                          double (JPEG_image);
  
               [W, H]                    =                          size(I);
        
               bpp                       =                          JPEG_info.FileSize*8/(W*H);       
              
              JPEG_header_info           =                          jpeg_read([jpeg_name '.jpg']);
              
              Par.QTable                 =                          JPEG_header_info.quant_tables{1}; 
        
              Par.blockSize              =                          8;
              
              Par.C_q                    =                          blkproc(Par.nim , [8 8], 'dct2'); 
        
              meanQuant                  =                          mean(mean(Par.QTable(1:3,1:3)));
              
              Par.nSig                   =                          1.195.*meanQuant.^0.6394 + 0.9673; %ндов╧╚й╫6
                            


        
 %--------------------------RRC Operator------------------------------------------%             
                  
               
             im                          =                      RRC_Deblocking_Main(Par.nim, Par);    
                              
                
              PSNR_Final                 =                          csnr (im, double(I), 0, 0);
              
              PSNR_B                     =                              compute_psnrb (im, double(I));

              FSIM_Final                 =                          cal_ssim (im, double(I), 0, 0);
              
              SSIM_Final                 =                          FeatureSIM (im, double(I));            
              
              
               if kk  ==3
              
               x_2_yuv(:,:,1)            =                          uint8(im);
               
            RRC_Result                   =                          ycbcr2rgb(uint8(x_2_yuv));   
              end
                           
              

if JPEG_Quality==10

Final_deblock= strcat(filename,'_RRC_BSD_','_JQ_',num2str(JPEG_Quality),'_PSNR_',num2str(PSNR_Final),'_PSNR_B_',num2str(PSNR_B),'_FSIM_',num2str(FSIM_Final), '_SSIM_',num2str(SSIM_Final),'.png');

imwrite(uint8(RRC_Result),strcat('./JQ_10_Result/',Final_deblock));





elseif JPEG_Quality==15

Final_deblock= strcat(filename,'_RRC_BSD_','_JQ_',num2str(JPEG_Quality),'_PSNR_',num2str(PSNR_Final),'_PSNR_B_',num2str(PSNR_B),'_FSIM_',num2str(FSIM_Final), '_SSIM_',num2str(SSIM_Final),'.png');
imwrite(uint8(RRC_Result),strcat('./JQ_15_Result/',Final_deblock));




elseif JPEG_Quality==20

Final_deblock= strcat(filename,'_RRC_BSD_','_JQ_',num2str(JPEG_Quality),'_PSNR_',num2str(PSNR_Final),'_PSNR_B_',num2str(PSNR_B),'_FSIM_',num2str(FSIM_Final), '_SSIM_',num2str(SSIM_Final),'.png');
imwrite(uint8(RRC_Result),strcat('./JQ_20_Result/',Final_deblock));



elseif JPEG_Quality==30

Final_deblock= strcat(filename,'_RRC_BSD_','_JQ_',num2str(JPEG_Quality),'_PSNR_',num2str(PSNR_Final),'_PSNR_B_',num2str(PSNR_B),'_FSIM_',num2str(FSIM_Final), '_SSIM_',num2str(SSIM_Final),'.png');

imwrite(uint8(RRC_Result),strcat('./JQ_30_Result/',Final_deblock));




elseif JPEG_Quality==40
    
Final_deblock= strcat(filename,'_RRC_BSD_','_JQ_',num2str(JPEG_Quality),'_PSNR_',num2str(PSNR_Final),'_PSNR_B_',num2str(PSNR_B),'_FSIM_',num2str(FSIM_Final), '_SSIM_',num2str(SSIM_Final),'.png');

imwrite(uint8(RRC_Result),strcat('./JQ_40_Result/',Final_deblock));


elseif JPEG_Quality==50
    
Final_deblock= strcat(filename,'_RRC_BSD_','_JQ_',num2str(JPEG_Quality),'_PSNR_',num2str(PSNR_Final),'_PSNR_B_',num2str(PSNR_B),'_FSIM_',num2str(FSIM_Final), '_SSIM_',num2str(SSIM_Final),'.png');


imwrite(uint8(RRC_Result),strcat('./JQ_50_Result/',Final_deblock));



elseif JPEG_Quality==60
    
Final_deblock= strcat(filename,'_RRC_BSD_','_JQ_',num2str(JPEG_Quality),'_PSNR_',num2str(PSNR_Final),'_PSNR_B_',num2str(PSNR_B),'_FSIM_',num2str(FSIM_Final), '_SSIM_',num2str(SSIM_Final),'.png');


imwrite(uint8(RRC_Result),strcat('./JQ_60_Result/',Final_deblock));



elseif JPEG_Quality==70
    
Final_deblock= strcat(filename,'_RRC_BSD_','_JQ_',num2str(JPEG_Quality),'_PSNR_',num2str(PSNR_Final),'_PSNR_B_',num2str(PSNR_B),'_FSIM_',num2str(FSIM_Final), '_SSIM_',num2str(SSIM_Final),'.png');

imwrite(uint8(RRC_Result),strcat('./JQ_70_Result/',Final_deblock));




elseif JPEG_Quality==80
    
Final_deblock= strcat(filename,'_RRC_BSD_','_JQ_',num2str(JPEG_Quality),'_PSNR_',num2str(PSNR_Final),'_PSNR_B_',num2str(PSNR_B),'_FSIM_',num2str(FSIM_Final), '_SSIM_',num2str(SSIM_Final),'.png');

imwrite(uint8(RRC_Result),strcat('./JQ_80_Result/',Final_deblock));


else
    
    

Final_deblock= strcat(filename,'_RRC_BSD_','_JQ_',num2str(JPEG_Quality),'_PSNR_',num2str(PSNR_Final),'_PSNR_B_',num2str(PSNR_B),'_FSIM_',num2str(FSIM_Final), '_SSIM_',num2str(SSIM_Final),'.png');

imwrite(uint8(RRC_Result),strcat('./JQ_90_Result/',Final_deblock));








end              
              
end

