

function  X  =   RRC_CORE( A, B, c1, nSig,ee)

[S, A0, D]            =                svd(full(A),'econ');  %svd(CurArray);

[m, ~]                =                size (A0);

[~, B0, ~]            =                svd(full(B),'econ');  %svd(CurArray);
 
 s0                   =                A0 -    B0;

 s0                   =                mean (s0.^2,2);
 
 s0                   =                max  (0, s0-nSig^2);
        
 lam                  =                repmat(((c1*sqrt(2)*nSig^2)./(sqrt(s0) + ee)),[1,m]);  % Eq. (14)
 
Alpha                 =                soft (A0-B0, lam)+ B0; % Eq. (12)
 
X                     =                S*Alpha*D';


return;