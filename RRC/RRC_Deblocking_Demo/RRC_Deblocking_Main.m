function [im, dif, j]                    =                          RRC_Deblocking_Main(Im_out, Par)


ori_im                      =                        Par.I; 

n_im                        =                        Im_out; 

[h1, w1]                    =                        size(ori_im); 


d_im                        =                        Im_out;  % 256 *256 初始化


v                           =                        Par.nSig; %噪声水平  18.3017


cnt                         =                         1;


All_PSNR                    =                         zeros(1,Par.Iter);

 RRC_Deback                 =                         cell (1,Par.Iter);   
    


for j                        =                1 : Par.Iter
    
       if (j ==1)
        d_im                 =                         n_im;
        
        nSig1                =                           v;
       end
     
      Par.nSig               =                         nSig1;
      
      
       d_im                  =                        RRC_Solver(d_im, Par);
       
       
        dif1                 =                         d_im-n_im;
        
    
        vd1                  =                          v^2-(mean(mean(dif1.^2)));
        
    
        nSig1                =                          sqrt(abs(vd1))* Par.lambda;
        
    
        d_im                 =                          (d_im*v^2*2*5+ n_im*nSig1^2)/(v^2*2*5 + nSig1^2);
     
     %    d_im                 =                          (d_im*v^2*2*Aa+ n_im*nSig1^2)/(v^2*2*Aa + nSig1^2);
    
        d_im                 =                           BDCT_project_onto_QCS(d_im, Par.C_q, Par.QTable, Par.Qfactor, Par.blockSize);
        
        RRC_Deback{j}        =                           d_im;
    
        
        All_PSNR(j)        =                            csnr( d_im(1:h1,1:w1), ori_im, 0, 0 );
    
      fprintf( 'Preprocessing, Iter %d : PSNR = %2.2f\n', cnt,  csnr( d_im(1:h1,1:w1), ori_im, 0, 0 ));
    
         cnt   =  cnt + 1;       
         

   
        if  j >5
      
             dif      =  norm(abs(RRC_Deback{j}) - abs(RRC_Deback{j-1}),'fro')/norm(abs(RRC_Deback{j-1}), 'fro');
       
         if dif< Par.error
           
               break;
         end
       
        end        
         
    
    
end


        im               =                 d_im;

end

