function par = setParameters(nSig)

par.downScaleFactor  = [1, 0.75]; %par.downScaleFactor  = [1, 0.75];

par.nSig = nSig;              % Noise variance of the input noisy image
par.SearchWinSize = 30;       % Nonlocal search domain for similar patches
par.Innerloop = 2;            % InnerLoop Num of between re-blockmatching
par.winSizeMethod='Fixed';
if nSig <= 20
    par.patsize = 7;         % patch size
    par.patnum = 70;         % the number of initial nonlocal similar pathces
    par.Iter = 8;            % total iter numbers
    par.lamada = 0.85;       % noise parameter for each interation
    par.delta = 0.10; 
elseif nSig <= 40
    par.patsize = 8;        
    par.patnum = 90;         
    par.Iter = 10;           
    par.lamada = 0.80;      
    par.delta = 0.10;       
elseif nSig <= 60
    par.patsize = 8;        
    par.patnum = 120;
    par.Iter = 10;
    par.lamada = 0.75;      
    par.delta = 0.05;  
else
    par.patsize = 8;       
    par.patnum = 140; 
    par.Iter = 14; 
    par.lamada = 0.65;     
    par.delta = 0.05; 
end

par.step =  floor((par.patsize)/2-1);

