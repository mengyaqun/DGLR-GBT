clc;
clear;

Y=double(imread("01.png"));
options.WeightMode = 'HeatKernel';options.t = 1;
Wr = similarity(Y, 'cosine'); 
Wc = full(constructW(Y', options));%%Wr = constructW(Y, options);
Lc = graph_laplacian(Wc);% Lr = graph_laplacian(Wr);