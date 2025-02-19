
function  Par   =    Par_Set(  I,  JPEG_Quality)

randn ('seed',0);

Par.I         =   double(I);

Par.Iter      =   20;

Par.eps       =   0.2;

Par.hp        =   40;  

Par.win       =   7;

Par.nblk      =   60;  

 Par.Qfactor  =   0.2;

if JPEG_Quality <= 1

    
    
    
    Par.lambda            =      0.3;   

    Par.c                 =      0.3;  
    
    Par.gamma             =      0.1;  
    
    Par.error             =     0.0015;
 
    
elseif JPEG_Quality <= 5

    
    Par.lambda            =      0.3;   

    Par.c                 =      0.9;  
    
   Par.gamma              =      0.1;  
   
   Par.error              =     0.0008; 
    
    
elseif JPEG_Quality <= 10

    
    Par.lambda            =      0.3;   

    Par.c                 =      0.9;  
    
    Par.gamma             =      0.1;   
    
    Par.error              =     0.0007; 

    
 elseif JPEG_Quality <= 15
        
    
    Par.lambda            =      0.3;   

    Par.c                 =      1.1;  
    
    Par.gamma             =      0.1;   
    
    Par.error              =     0.0005; 
    
 elseif JPEG_Quality <= 20

    
    Par.lambda            =      0.2;   

    Par.c                 =      1.3;  
    
    Par.gamma            =      0.1;   
    
    Par.error              =     0.0005;     
    
 elseif JPEG_Quality <=  30

    
    Par.lambda            =      0.2;   

    Par.c                 =      1.3;  
    
    Par.gamma             =      0.1; 
    
    Par.error              =     0.0003;   

else
    
    
    Par.lambda            =      0.2;   

    Par.c                 =      1.5;  
    
    Par.gamma             =      0.1;   
   
    Par.error             =    0.0003;   
    
end

Par.step      =   min(4, Par.win-1);

end

