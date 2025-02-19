 
function [Denoising,  iter,Err_or]   =  LR_GSC_Denoising( par)

randn ('seed',0);

Nim                =   par.nim;

Ori_im             =   par.I;

b                  =   par.win;

[h, w, ch]         =   size(Nim);

N                  =   h-b+1;

M                  =   w-b+1;

r                  =   [1:N];

c                  =   [1:M]; 

disp(sprintf('PSNR of the noisy image = %f \n', csnr(Nim, Ori_im, 0, 0) ));
disp(sprintf('SSIM of the noisy image = %f \n', cal_ssim(Nim, Ori_im,0,0) )); 
Im_Out      =   Nim;

nsig        =   par.nSig;

m           =   par.nblk;%KNN    

cnt         =   1;


AllPSNR     =  zeros(1,par.Iter );

Denoising  =   cell (1,par.Iter);

for iter = 1 : par.Iter    
    
            Im_Out               =   Im_Out + par.gamma*(Nim - Im_Out);
        
            dif                  =   Im_Out-Nim;
        
            vd                   =      nsig^2-(mean(mean(dif.^2)));
        
       if iter==1
            
            par.nSig             =         sqrt(abs(vd)); 
            
       else
            
             par.nSig            =         sqrt(abs(vd))*par.lamada;
            
       end 

              blk_arr           =          Block_matching( Im_Out, par);  %            Ϊ4 £ ÿ            60     ƿ λ      

      
        X                       =         Im2Patch( Im_Out, par );  %        
        
        Ys                      =         zeros( size(X) );    
        
        W                       =         zeros( size(X) );
        
        K                       =         size(blk_arr,2);
           
    
        
        for  i  =  1 : K  
            
            
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %Noise_Group_Operator...
             B                  =         X(:, blk_arr(:, i));  %  i   60     ƿ 
             
             mB                 =         repmat(mean( B, 2 ), 1, size(B, 2));%           ƽ  ֵ
             
             curArray           =         B-mB;   %Ϊ           õļ   
            
               U_i              =         getpca(curArray); % generate PCA basis
        
               A0               =         U_i'*curArray;       
             
             
    % B_i subproblem              
             
          [GR_S, GR_V, GR_D]    =                 svd(full(A0),'econ');  %svd(CurArray);    
          
          

          
          cc                    =                 sqrt( mean(GR_V.^2, 2) );
          
          lambda_1              =                 2*sqrt(2)*par.c2*par. mu*par.nSig^2./(cc + eps);
          
          Thre                  =                 1./(diag(GR_V)+  0.25);
          
          GR_Z                  =                 soft(GR_V,diag(lambda_1.*Thre)); 
          
           BG                   =                 GR_S*GR_Z*GR_D'; 
           
           
  % A_i subproblem
   
         QG                     =                (par.mu*par.nSig^2*A0 + nsig^2*BG)/ (par.mu* par.nSig^2 + nsig^2+ eps);
         


         s0                     =                 mean (QG.^2,2);

         s0                     =                 max  (0, s0 - par.nSig^2);    
            
         lambda_2               =                 repmat ( 2*sqrt(2)*par. c1 *par. mu*par.nSig^2*nsig^2./(sqrt(s0)+eps),[1, size(QG,2)]); %Generate the weight Eq.(19)
        
         Alpha                  =                 soft (QG, lambda_2);  

         TMP                    =                 U_i*Alpha +  mB;          
           
          
             
             
           
       Ys(:, blk_arr(1:m,i))    =          Ys(:, blk_arr(1:m,i)) + TMP;
       
       W(:, blk_arr(1:m,i))     =          W(:, blk_arr(1:m,i)) + 1;
             
         end

        Im_Out   =  zeros(h,w);
        
        im_wei   =  zeros(h,w);
        
        k        =   0;
        
        for i  = 1:b
            for j  = 1:b
                k    =  k+1;
                Im_Out(r-1+i,c-1+j)  =  Im_Out(r-1+i,c-1+j) + reshape( Ys(k,:)', [N M]);
                im_wei(r-1+i,c-1+j)  =  im_wei(r-1+i,c-1+j) + reshape( W(k,:)',  [N M]);
            end
        end
        
        Im_Out  =  Im_Out./(im_wei+eps);
        
        Denoising{iter}  =   Im_Out;

        AllPSNR(iter)  = csnr( Im_Out, par.I, 0, 1 );
              
        fprintf( 'LR_GSC Denoising:  Iteration %d : nSig = %2.2f, PSNR = %f\n', cnt, par.nSig, csnr( Im_Out, par.I, 0, 1) );
        
        cnt   =  cnt + 1;

   if iter>1
       
        Err_or      =  norm(abs(Denoising{iter}) - abs(Denoising{iter-1}),'fro')/norm(abs(Denoising{iter-1}), 'fro');
         
       
       if AllPSNR(iter) - AllPSNR(iter-1) <0
           break;
       end    
   end    
   
end
im      =      Denoising{iter-1};

disp(sprintf('PSNR of the Denoised image = %f \n', csnr(im, Ori_im, 0, 0) ));
disp(sprintf('SSIM of the Denoised image = %f \n', cal_ssim(im, Ori_im, 0, 0) ));
return;




