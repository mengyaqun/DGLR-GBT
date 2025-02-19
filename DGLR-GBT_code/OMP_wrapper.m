function OMP_wrapper(dict_path, signal_path, out_name, t_sparsity)
%=============================================
% Wrapper for OMP
% Convenience function for Pan et al., 2021
%{
首先，定义一个名为OMP_wrapper的函数，它接受四个参数：dict_path（字典路径），signal_path（信号路径），
out_name（输出文件名）和t_sparsity（稀疏度）。
rng('default')% 设置随机数生成器的种子，以便在每次运行时得到相同的结果
dict = readmatrix(dict_path);% 读取字典文件
dict = normcols(dict);% 对字典的每一列进行归一化处理
signal = readmatrix(signal_path);% 读取信号文件
g_mat = dict'*dict;% 计算字典与自身的转置矩阵
out = omp(dict, signal, g_mat, t_sparsity);% 使用 OMP 算法对信号进行稀疏分解
writematrix(out, out_name);% 将稀疏分解结果写入文件
end
在 MATLAB 中，这段代码定义了一个名为OMP_wrapper的函数，用于对给定的信号进行字典学习（OMP 算法）和稀疏分解。
该函数首先读取字典文件和信号文件，然后计算字典的转置矩阵。接着，使用 OMP 算法对信号进行稀疏分解，并将结果保存到指定的输出文件中。
%}
rng('default')

dict = readmatrix(dict_path);

dict = normcols(dict);
signal = readmatrix(signal_path);

g_mat = dict'*dict;

out = omp(dict, signal, g_mat, t_sparsity);

writematrix(out, out_name);