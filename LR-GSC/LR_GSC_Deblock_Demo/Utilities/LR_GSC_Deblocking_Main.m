function    [im, j, diff]                    =                          LR_GSC_Deblocking_Main(Im_out, Opts)

   randn ('seed',0);
   
ori_im                      =                        Opts.I; 

n_im                        =                        Im_out; 

[h1, w1]                    =                        size(ori_im); 


d_im                        =                        Im_out;  % 256 *256 初始化


v                           =                        Opts.nSig; %噪声水平  18.3017


B                           =                       zeros (size (n_im));

mu                          =                        Opts. mu;


cnt                         =                         1;


All_PSNR                    =                         zeros(1,Opts.Iter);

EN_SSP_Results             =                         cell (1,Opts.Iter);   


for j                        =                1 : Opts.Iter
    
       %if (j ==1)
     %   d_im                 =                         n_im;
        
     %   nSig1                =                           v;
    %   end
    
      
      
  %     d_im                  =                        SSR_ARC_Solver(d_im, Deno, p);
  
          d_im                 =   d_im +    0.1 *(n_im - d_im);
  
  
          dif1                 =                         d_im-n_im;
        
    
          vd                   =                          v^2-(mean(mean(dif1.^2)));
        
        
       if j==1
            
            Opts.nSig             =         sqrt(abs(vd)); 
            
       else
            
             Opts.nSig            =         sqrt(abs(vd))* Opts.lambda  ;
            
       end 
       
       
       

          Inter_out          =                     LR_GSC_Core  (d_im, Opts); 
          
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ADMM%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
          
          L                   =                     ( Opts.alpha*  Opts.nSig^2* d_im -  Opts.alpha *  Opts.nSig^2* B + mu * Inter_out)/ ( Opts.alpha *  Opts.nSig^2 + mu);
          
          
          
          d_im                =          (mu* Opts.nim + v^2 * (L + B))/  (mu + v^2);
          
 
    
        d_im                   =                           BDCT_project_onto_QCS(d_im, Opts.C_q, Opts.QTable, Opts.Qfactor, Opts.blockSize);
    
            B                  =                            B  - (d_im - L);
            
        All_PSNR(j)             =                          csnr( d_im(1:h1,1:w1), ori_im, 0, 0 );
    
      fprintf( 'LR_GSC Debloking:  Preprocessing, Iter %d : PSNR = %2.2f\n', cnt,  csnr( d_im(1:h1,1:w1), ori_im, 0, 0 ));
    
         cnt   =  cnt + 1;       
         
           EN_SSP_Results{j}      =                      d_im;
         
         if j>3
             
            diff      =  norm(abs(EN_SSP_Results{j}) - abs(EN_SSP_Results{j-1}),'fro')/norm(abs(EN_SSP_Results{j-1}), 'fro');    
             
             if      All_PSNR(j) -  All_PSNR(j-1)  <0 %diff<= Err_or
                 
                 break;
             end
             
         end
       
      
    
    
end


   %     im               =                 d_im;
        
        im               =                     EN_SSP_Results{j-1};                                      

end

