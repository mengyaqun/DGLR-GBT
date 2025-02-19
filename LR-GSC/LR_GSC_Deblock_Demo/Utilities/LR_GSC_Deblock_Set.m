
function  [ Opts]   =    LR_GSC_Deblock_Set(   I, JPEG_Quality, lambda, alpha, beta, mu,  c1, c2)

randn ('seed',0);

Opts.I         =   double(I);

Opts.Iter      =   45;

Opts.step      =   4;         

Opts.patch     =  7;   

Opts.Sim       =  40;  


if JPEG_Quality <= 1
      
    Opts.lambda      =  lambda;   
    
    Opts. alpha      =  alpha;    
    
    Opts. beta       =  beta; 
    
    Opts. mu         =  mu;     
    
    Opts. c1         =   c1;   

    Opts. c2         =   c2;    
    
elseif JPEG_Quality <= 5
    
    Opts.lambda      =  lambda;   
    
    Opts. alpha      =  alpha;    
    
    Opts. beta       =  beta; 
    
    Opts. mu         =  mu;     
    
    Opts. c1         =   c1;   

    Opts. c2         =   c2;    
     
    
elseif JPEG_Quality <= 10
    
    Opts.lambda      =  lambda;   
    
    Opts. alpha      =  alpha;    
    
    Opts. beta       =  beta; 
    
    Opts. mu         =  mu;     
    
    Opts. c1         =   c1;   

    Opts. c2         =   c2;       
    
    
 elseif JPEG_Quality <= 20
    
    Opts.lambda      =  lambda;   
    
    Opts. alpha      =  alpha;    
    
    Opts. beta       =  beta; 
    
    Opts. mu         =  mu;     
    
    Opts. c1         =   c1;   

    Opts. c2         =   c2;       
    
elseif JPEG_Quality <= 30
    
    Opts.lambda      =  lambda;   
    
    Opts. alpha      =  alpha;    
    
    Opts. beta       =  beta; 
    
    Opts. mu         =  mu;     
    
    Opts. c1         =   c1;   

    Opts. c2         =   c2;     

elseif JPEG_Quality <= 40
    
    Opts.lambda      =  lambda;   
    
    Opts. alpha      =  alpha;    
    
    Opts. beta       =  beta; 
    
    Opts. mu         =  mu;     
    
    Opts. c1         =   c1;   

    Opts. c2         =   c2;      
    
else
    
    Opts.lambda      =  lambda;   
    
    Opts. alpha      =  alpha;    
    
    Opts. beta       =  beta; 
    
    Opts. mu         =  mu;     
    
    Opts. c1         =   c1;   

    Opts. c2         =   c2;    
    
end


end

