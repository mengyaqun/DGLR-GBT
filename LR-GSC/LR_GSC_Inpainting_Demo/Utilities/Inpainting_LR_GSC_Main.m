

function [reconstructed_image, PSN_Result,FSIM_Result, SSIM_Result, All_PSNR, j, Err_or] = Inpainting_LR_GSC_Main(Y,Opts)


Initial                             =                       Opts.initial;

mu                                  =                       Opts.mu;

IterNums                            =                       Opts.IterNums;

X_Org                               =                       Opts.org;

MASK                                =                       Opts.A;

A                                   =                       @(x) MASK.*x;

AT                                  =                       @(x) MASK.*x;

ATy                                 =                        AT(Y);

d_im                                =                        Initial;

B                                   =                        zeros(size(Y));

%L                                   =                        zeros(size(Y));

Muinv                               =                         1/mu;

InvAAT                              =                        1./(mu+MASK);

All_PSNR                            =                        zeros(1,IterNums);

 BRM_Results                        =                        cell (1,IterNums);   

fprintf('Initial PSNR = %f\n',csnr(d_im,X_Org,0,0));

for j = 1:IterNums
    
   
   Inter_out                        =                     LR_GSC_Core  (d_im, Opts); 
   
   

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ADMM%%%%%%%%%%%%%%%%%%%%%%
   
       L                            =                      (mu*d_im-mu*B + Opts.alpha* Inter_out)/ (mu+ Opts.alpha);
       
       R                            =                       ATy +mu*(L+B); 
    
     d_im                           =                       Muinv*(R - AT(InvAAT.*A(R)));  
    
     B                              =                        B + (L - d_im);   
    
    All_PSNR(j)                     =                        csnr(d_im,X_Org,0,0);
    
    BRM_Results{j}            =                        d_im;
    
    fprintf('iter number = %d, PSNR = %f, SSIM = %f\n',j,csnr(d_im,X_Org,0,0),cal_ssim(X_Org,d_im,0,0));

    if j>3
        
             Err_or      =  norm(abs(BRM_Results{j}) - abs(BRM_Results{j-1}),'fro')/norm(abs(BRM_Results{j-1}), 'fro');

        if(All_PSNR(j)-All_PSNR(j-1)<0)
            
            break;
            
        end
    end

end

reconstructed_image = d_im;

PSN_Result  = csnr(reconstructed_image,X_Org,0,0);
FSIM_Result = FeatureSIM(reconstructed_image,X_Org);
SSIM_Result = cal_ssim(reconstructed_image,X_Org,0,0);


end

