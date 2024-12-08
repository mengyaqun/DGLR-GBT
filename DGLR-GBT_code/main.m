% 目的：执行 DGRL-GBT 去噪 
% 输入：Y：噪声图像 
%       x：用于计算 PSNR 的原始图像 
%       nSig：噪声方差 
% 输出：im_out：去噪图像 
%       psnr：去噪图像的 PSNR 

clc;
clear;
PSNR=zeros(400);    %记录每次迭代的PSNR值
SSIM=zeros(400);    %记录每次迭代的SSIM值
Psnr=zeros(30);     %只记录最终去噪图像的PSNR
Ssim=zeros(30);     %只记录最终去噪图像的SSIM

for v=3:4   %选取测试图像
    if v==1
        im_in = double(imread('D:\image desioning\MATLAB\DGLR-GBT\images\cones.png'));	%图像路径
    elseif v==2
        im_in = double(imread('D:\image desioning\MATLAB\DGLR-GBT\images\baby.png'));
    elseif v==3
        im_in = double(imread('D:\image desioning\MATLAB\DGLR-GBT\images\aloe.png'));   
    elseif v==4
        im_in = double(imread('D:\image desioning\MATLAB\DGLR-GBT\images\lampshade.png'));
    elseif v==5
        im_in = double(imread('D:\image desioning\MATLAB\DGLR-GBT\images\moebius.png'));
    elseif v==6
        im_in = double(imread('D:\image desioning\MATLAB\DGLR-GBT\images\reindeer.png'));
    elseif v==7
        im_in = double(imread('D:\image desioning\MATLAB\DGLR-GBT\images\dolls.png'));
    elseif v==8
        im_in = double(imread('D:\image desioning\MATLAB\DGLR-GBT\images\baby2.png'));
    elseif v==9
        im_in = double(imread('D:\image desioning\MATLAB\DGLR-GBT\images\baby3.png'));
    elseif v==10
        im_in = double(imread('D:\image desioning\MATLAB\DGLR-GBT\images\plastic.png'));
    elseif v==11
        im_in = double(imread('D:\image desioning\MATLAB\DGLR-GBT\images\wood.png'));
    elseif v==13
        im_in = double(imread('D:\image desioning\MATLAB\DGLR-GBT\images\bowling.png'));
    elseif v==14
        im_in = double(imread('D:\image desioning\MATLAB\DGLR-GBT\images\lampshade2.png'));
    elseif v==15
        im_in = double(imread('D:\image desioning\MATLAB\DGLR-GBT\images\laundry.png'));
    elseif v==16
        im_in = double(imread('D:\image desioning\MATLAB\DGLR-GBT\images\monopoly.png'));   
    elseif v==17
        im_in = double(imread('D:\image desioning\MATLAB\DGLR-GBT\images\books.png'));
    elseif v==18
        im_in = double(imread('D:\image desioning\MATLAB\DGLR-GBT\images\baby.png'));
    elseif v==19
        im_in = double(imread('D:\image desioning\MATLAB\DGLR-GBT\images\bowling2.png'));
    elseif v==20
        im_in = double(imread('D:\image desioning\MATLAB\DGLR-GBT\images\midd2.png'));
    elseif v==21
        im_in = double(imread('D:\image desioning\MATLAB\DGLR-GBT\images\wood2.png'));
    elseif v==22
        im_in = double(imread('D:\image desioning\MATLAB\DGLR-GBT\images\teddy.png'));
    elseif v==23
        im_in = double(imread('D:\image desioning\MATLAB\DGLR-GBT\images\rocks.png'));
    elseif v==24
        im_in = double(imread('D:\image desioning\MATLAB\DGLR-GBT\images\rocks2.png'));
    elseif v==25
        im_in = double(imread('D:\image desioning\MATLAB\DGLR-GBT\images\cones.png'));
    elseif v==26
        im_in = double(imread('D:\image desioning\MATLAB\DGLR-GBT\images\midd.png'));
    elseif v==27
        im_in = double(imread('D:\image desioning\MATLAB\DGLR-GBT\images\flowerpots.png'));
    elseif v==28
        im_in = double(imread('D:\image desioning\MATLAB\DGLR-GBT\images\cloth.png'));
    elseif v==29
        im_in = double(imread('D:\image desioning\MATLAB\DGLR-GBT\images\cloth2.png'));
    elseif v==30
        im_in = double(imread('D:\image desioning\MATLAB\DGLR-GBT\images\cloth3.png'));
    elseif v==31
        im_in = double(imread('D:\image desioning\MATLAB\DGLR-GBT\images\cloth4.png'));
    else 
        break;
    end
    
    
    noisy =[10,15,20,25,30,40,50];  % noise variance
    
    for n=1:length(noisy)
        nSig = noisy(n);
        im_noisy = im_in + randn(size(im_in)) * nSig; % generate the noisy image
        y=im_noisy;
        x=im_in;
        im_out = y;
        im_out2 = y;
        [h, w] = size(y);
        par.nSig = nSig;
    
        % parameters
            wth=1; 		%初始化
        if nSig <= 10
            par.win       =   5;      % window size 窗口大小
            par.nblk      =   25;     % number of K-nearest-neighbors K最近临近
            par.lamada    =   0.63;   % parameter for noise variance update噪声方差更新参数
            par.w         =   0.13;   % parameter for iterative regularization用于迭代正则化参数
            times         =   6;

        elseif nSig <= 15
            par.win       =   5;
            par.nblk      =   25;
            par.lamada    =   0.63;
            par.w         =   0.13;
            times         =   7;
           
        elseif nSig <= 20
            par.win       =   5;
            par.nblk      =   25;
            par.lamada    =   0.63;
            par.w         =   0.08;
            times         =   9;
            
        elseif nSig <= 25
            par.win       =   6;
            par.nblk      =   30;
            par.lamada    =   0.63;
            par.w         =   0.10;
            times         =   8;
           
        elseif nSig <= 30
            par.win       =   6;
            par.nblk      =   30;
            par.lamada    =   0.65;
            par.w         =   0.08;
            times         =   9;

        elseif nSig <= 40
            par.win       =   7;
            par.nblk      =   50;
            par.lamada    =   0.65;
            par.w         =   0.08;
            times         =   12;

        elseif nSig <= 50
            par.win       =   8;
            par.nblk      =   60;
            par.lamada    =   0.67;
            par.w         =   0.08;
            times         =   16;

        else 
            par.win       =   9;
            par.nblk      =   80;
            par.lamada    =   0.67;
            par.w         =   0.08;
            times         =   18;
        end
        par.step = min(6, par.win-1);
        lambda = par.w;

        fprintf('选取第 %d 张图像进行测试，高斯白噪声强度为：%ddb \n', v, nSig);

        iter = 1; % 初次迭代
        % step 1: iterative regularization & noise variance update          
        % 第 1 步：迭代正则化和噪声方差更新
        im_out  =   im_out + lambda*(y - im_out);
            
        dif     =   im_out - y;
        vd      =   nSig^2 - (mean(mean(dif.^2)));
                
        if iter==1
            par.nSig  = sqrt(abs(vd));            
        else
            par.nSig  = sqrt(abs(vd))*par.lamada;
        end    
            
        % step 2: patch clustering
        % 第 2 步：补丁聚类
        if (mod(iter,6)==0) || (iter==1)
            blk_arr   =   Block_matching( im_out, par);
        end
        
        X  =  Im2Patch( im_out, par );    
        Ys =  zeros( size(X) );   
        W  =  zeros( size(X) );
        L = size(blk_arr,2);
        
        for  i  =  1 : L
    
            B = double(X(:, blk_arr(:, i)));
            
           %% step 3:  denoising
            % step 3.1: compute the average patch for a group of similar patches       
            
            % 计算每行去掉最大最小值后的平均值  
            mP = zeros(size(B, 1), 1);  
            for hh = 1:size(B, 1)  
                % 去掉每行的最大值和最小值  
                B3 = B(hh, B(hh,:) < max(B(hh,:)) & B(hh,:) > min(B(hh,:)));  
                % 计算平均值  
                mP(hh) = mean(B3);  
            end
            
            mblk = reshape(mP,par.win,par.win);
            
            % step 3.2: compute the weighted GBT basis functions based on the average patch 
            [V] = WGBT(mblk,par.nSig);
            
            % step 3.3: dual_admm
            options.WeightMode = 'HeatKernel';options.t = 1;
            Wc = constructW(B', options);
            Wr = constructW(B, options);
            Lc = graph_laplacian(Wc); 
            Lr = graph_laplacian(Wr);
            params = struct();
            params.Y = B;
            dict=V;
            dict = normcols(dict);
            g_mat = dict'*dict;
            params.D = V;
            params.T = 3;
            params.alpha =1; params.Lr = Lr;
            params.beta  =1;  params.Lc = Lc;
            params.Z = omp(V,B,g_mat,3);
            X1 = DUAL_ADMM(params);
            B2 = round(2*V*X1);
            bi=B./B2;
            
            % step 3.4: hard-thresholding
            
            A = V'*B2;    % the GBT coefficient matrix
            %加权阈值wth
            wth = (-0.01*nSig+0.95+0.05*sign(35-nSig))*0.5*(1-sign(nSig-50))+0.2*(1+sign(nSig-50));
            th = (par.nSig)*sqrt(2*log((par.win)^2*(par.nblk)))*wth;
            A(abs(A)<th) = 0;

            % step 3.4: reconstruction
            B_hat = round(V*A);       
            r = rank(A);
            if r==size(B2,1)
                wei = 1/size(B2,1);
            else
                wei = (size(B2,1)-r)/size(B2,1);
            end
            
            W(:, blk_arr(:,i)) = wei*ones( size(B_hat) );  
            Ys(:, blk_arr(:,i)) = (B_hat)*wei;
        end
    
        % step 4: image update
        im_out   =  zeros(h,w);
        im_wei   =  zeros(h,w);
        k        =   0;
        b        =   par.win;
        N        =   h-b+1;
        M        =   w-b+1;
        r        =   [1:N];
        c        =   [1:M]; 
        for i  = 1:b
            for j  = 1:b
                k    =  k+1;
                im_out(r-1+i,c-1+j)  =  im_out(r-1+i,c-1+j) + reshape( Ys(k,:)', [N M]);
                im_wei(r-1+i,c-1+j)  =  im_wei(r-1+i,c-1+j) + reshape( W(k,:)', [N M]);
            end
        end
        im_out  =  im_out./(im_wei+eps);      
        
        PSNR((v-1)*400+iter+1+(n-1)*40) = csnr( im_out, x, 0, 0 );
        SSIM((v-1)*400+iter+1+(n-1)*40) = ssim(im_out/256,im_in/256);
        fprintf('第 %d 次迭代结果为：PSNR=%.2f，SSIM=%.4f ;', iter,csnr( im_out, x, 0, 0 ),ssim(im_out/256,im_in/256));
        
        %根据初次去噪结果微调迭代次数
        % ssim(im_out/256,im_noisy/256)：初次迭代后图与原噪声图的SSIM值
        
        if(times==8 && ssim(im_out/256,im_noisy/256) <= 0.20)
            times=11;
        end
        if(times==9 && nSig>20 && ssim(im_out/256,im_noisy/256) <= 0.24)
            times=13;
        end
        if(times==12 && ssim(im_out/256,im_noisy/256) <= 0.4935 )
            times=15;
        end
        if(times==16 && ssim(im_out/256,im_noisy/256) <= 0.7210 )
            times=24;
        end
        fprintf('迭代总次数为： %d \n', times);

     for iter = 2 : times %开始循环迭代
            % step 1: iterative regularization & noise variance update          
            % 第 1 步：迭代正则化和噪声方差更新
            im_out  =   im_out + lambda*(y - im_out);
            
            dif     =   im_out - y;
            vd      =   nSig^2 - (mean(mean(dif.^2)));
                
            if iter==1
                par.nSig  = sqrt(abs(vd));            
            else
                par.nSig  = sqrt(abs(vd))*par.lamada;
            end    
            
            % step 2: patch clustering
            % 第 2 步：补丁聚类
            if (mod(iter,6)==0) || (iter==1)
                blk_arr   =   Block_matching( im_out, par);
            end
        
            X  =  Im2Patch( im_out, par );    
            Ys =  zeros( size(X) );   
            W  =  zeros( size(X) );
            L = size(blk_arr,2);
        
       for  i  =  1 : L
    
            B = double(X(:, blk_arr(:, i)));
            
           %% step 3:  denoising
            % step 3.1: compute the average patch for a group of similar patches       
            
            % 计算每行去掉最大最小值后的平均值  
            mP = zeros(size(B, 1), 1);  
            for hh = 1:size(B, 1)  
                % 去掉每行的最大值和最小值  
                B3 = B(hh, B(hh,:) < max(B(hh,:)) & B(hh,:) > min(B(hh,:)));  
                % 计算平均值  
                mP(hh) = mean(B3);  
            end
            
            mblk = reshape(mP,par.win,par.win);
            
            % step 3.2: compute the weighted GBT basis functions based on the average patch 
            [V] = WGBT(mblk,par.nSig);
            
            % step 3.3: dual_admm
            options.WeightMode = 'HeatKernel';options.t = 1;
            Wc = constructW(B', options);
            Wr = constructW(B, options);
            Lc = graph_laplacian(Wc); 
            Lr = graph_laplacian(Wr);
            params = struct();
            params.Y = B;
            dict=V;
            dict = normcols(dict);
            g_mat = dict'*dict;
            params.D = V;
            params.T = 3;
            params.alpha =1; params.Lr = Lr;
            params.beta  =1;  params.Lc = Lc;
            params.Z = omp(V,B,g_mat,3);
            X1 = DUAL_ADMM(params);
            B2 = round(2*V*X1);
            bi=B./B2;
            
            % step 3.4: hard-thresholding
            
            A = V'*B2;    % the GBT coefficient matrix
            %加权阈值wth
            wth = (-0.01*nSig+0.95+0.05*sign(35-nSig))*0.5*(1-sign(nSig-50))+0.2*(1+sign(nSig-50));
            th = (par.nSig)*sqrt(2*log((par.win)^2*(par.nblk)))*wth;
            A(abs(A)<th) = 0;

            % step 3.4: reconstruction
            B_hat = round(V*A);       
            r = rank(A);
            if r==size(B2,1)
                wei = 1/size(B2,1);
            else
                wei = (size(B2,1)-r)/size(B2,1);
            end
            
            W(:, blk_arr(:,i)) = wei*ones( size(B_hat) );  
            Ys(:, blk_arr(:,i)) = (B_hat)*wei;
        end
    
        % step 4: image update
        im_out   =  zeros(h,w);
        im_wei   =  zeros(h,w);
        k        =   0;
        b        =   par.win;
        N        =   h-b+1;
        M        =   w-b+1;
        r        =   [1:N];
        c        =   [1:M]; 
        for i  = 1:b
            for j  = 1:b
                k    =  k+1;
                im_out(r-1+i,c-1+j)  =  im_out(r-1+i,c-1+j) + reshape( Ys(k,:)', [N M]);
                im_wei(r-1+i,c-1+j)  =  im_wei(r-1+i,c-1+j) + reshape( W(k,:)', [N M]);
            end
        end
        im_out  =  im_out./(im_wei+eps);      
        
        PSNR((v-1)*400+iter+1+(n-1)*40) = csnr( im_out, x, 0, 0 );
        SSIM((v-1)*400+iter+1+(n-1)*40) = ssim(im_out/256,im_in/256);
        fprintf('第 %d 次迭代结果为：PSNR=%.2f，SSIM=%.4f\n', iter,csnr( im_out, x, 0, 0 ),ssim(im_out/256,im_in/256));
      
     end
    end
end