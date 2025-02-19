
function  Opts   =    Opts_Set( nSig,  I )

randn ('seed',0);

Opts.I         =   double(I);

Opts.nSig      =   nSig;

Opts.Iter      =   25;

Opts.eps       =   0.2;

Opts.hp        =   40;  


if nSig <= 10
    
    Opts.win       =   6;
    
    Opts.nblk      =   60;   
    
    Opts.c1        =  0.9;  
    
    Opts.gamma     =  0.1;     
    
    Opts.lamada    =  0.9;   
    
    Opts.error     =   0.001;   
    
elseif nSig <= 20
    
    Opts.win       =   6;
    
    Opts.nblk      =   60;   
    
    Opts.c1        =  0.9;  
    
    Opts.gamma     =  0.1;     
    
    Opts.lamada    =  0.9;

    Opts.error     =   0.001;   
     
    
elseif nSig <= 30
    
    Opts.win       =   7;
    
    Opts.nblk      =   60;
    
    Opts.c1        =   0.9;  
    
    Opts.gamma     =   0.1;     
    
    Opts.lamada    =   0.8;
    
    Opts.error     =   0.001;     
    
    
 elseif nSig <= 40
    
    Opts.win       =   7;
    
    Opts.nblk      =   70;
    
    Opts.c1        =   0.9;  
    
    Opts.gamma     =   0.1;     
    
    Opts.lamada    =   0.8;
    
    Opts.error     =   0.0006;   
    
elseif nSig<=50
    
    Opts.win       =   7;
    
    Opts.nblk      =   80;
     
    Opts.c1        =   1;  
    
    Opts.gamma     =   0.1;     
    
    Opts.lamada    =   0.8;
    
    Opts.error     =   0.0006;     

elseif nSig<=75
    
    Opts.win       =   8;
    
    Opts.nblk      =   90;
    
    Opts.c1        =   1;  
    
    Opts.gamma     =   0.1;     
    
    Opts.lamada    =   0.8;
    
    Opts.error     =   0.0005;     
    
else
    
    Opts.win       =   9; 
    
    Opts.nblk      =   100;
    
    Opts.c1        =   1;  
    
    Opts.gamma     =   0.1;     
    
    Opts.lamada    =   0.8;
    
    Opts.error     =   0.002;     
end

Opts.step      =   min(4, Opts.win-1);

end

