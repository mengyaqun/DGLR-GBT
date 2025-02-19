% 读取 RGB 图像

img = imread('D:\image desioning\MATLAB\RGBZ\215.png');  % 替换为你的图像路径

% 将 RGB 图像转换为灰度图像
gray_img = rgb2gray(img);

% 显示灰度图像（可选）
imshow(gray_img, []);
title('Gray Image');

% 保存灰度图像到指定路径
output_path = 'D:\image desioning\MATLAB\RGBZ\015.png';  % 指定保存路径和文件名
imwrite(gray_img, output_path);

% 提示保存成功
disp(['Gray image saved to ', output_path]);