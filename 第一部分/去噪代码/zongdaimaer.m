clear;
clc;
% --- 目标保存路径设置 ---
save_folder = 'C:\Users\HUAWEI\Desktop\数值方法大作业资料包\第一部分\去噪结果1\'; % 目标文件夹
save_filename = '去噪结果图.png'; % 目标文件名
full_save_path = fullfile(save_folder, save_filename);
% -------------------------

% 读取图像
% 请确保路径正确
img1 = imread('C:\Users\HUAWEI\Desktop\数值方法大作业资料包\第一部分\同一图像不同噪声强度\原始图像.png');
img = imread('C:\Users\HUAWEI\Desktop\数值方法大作业资料包\第一部分\同一图像不同噪声强度\原始图像.png');
% 转换为灰度-双精图像
img_grey = convertToGray(img);
original_img = convertToGray(img1);
original_img = double(original_img);
img_grey = double(img_grey);
[m , n] = size(img_grey);
% 参数设置
mu = 250;
lambda = 100;  
max_iter = 10000;
tolerance = 1e-4;
% 初始化变量
real_image = img_grey;
real_image_prev = zeros(m, n);
d_xx = zeros(m, n);
d_xy = zeros(m, n);
d_yy = zeros(m, n);
b_xx = zeros(m, n);
b_xy = zeros(m, n);
b_yy = zeros(m, n);
% 【优化2】预分配收敛记录数组，避免动态内存分配
change_list = zeros(max_iter, 1);
% 【优化1】将算子构造和固定项计算移出循环
% 预计算频域算子求解 (只计算一次)
[H_xx, H_yy, H_xy, denom] = setup_frequency_operators(m, n, mu, lambda);
% 预计算图像的FFT以及分子的固定部分
v_img_grey = fft2(img_grey);
numerator_part1 = (mu/lambda) * v_img_grey; 
% 迭代优化
tic; % 简单计时
final_iter = 0;
for iter = 1:max_iter
    real_image_prev = real_image;
    
    % 更新 f(在频域中求解)
    % 注意：这里传入 numerator_part1 稍微修改了一下 update_f 的接口以利用预计算值
    real_image = update_f_optimized(numerator_part1, d_xx, d_xy, d_yy, ...
                         b_xx, b_xy, b_yy, H_xx, H_yy, H_xy, denom);
    
    % 计算梯度
    [real_image_xx, real_image_yy, real_image_xy] = compute_gradients(real_image, H_xx, H_yy, H_xy);
    
    % 更新辅助变量 d
    [d_xx, d_yy, d_xy] = update_d(real_image_xx, real_image_yy, real_image_xy, ...
                                 b_xx, b_yy, b_xy, lambda);
    
    % 更新 Bregman 变量 b
    [b_xx, b_yy, b_xy] = update_b(real_image_xx, real_image_yy, real_image_xy, ...
                                     d_xx, d_yy, d_xy, b_xx, b_yy, b_xy);
    
    % 检查收敛 (修改了调用方式以适应预分配)
    [converged, relative_change] = check_convergence_val(real_image, real_image_prev, tolerance, iter);
    
    % 记录误差
    change_list(iter) = relative_change;
    final_iter = iter;
    disp(relative_change);
    if converged
        fprintf('收敛于迭代 %d\n', iter);
        break;
    end
    if mod(iter, 100) == 0
        fprintf('迭代 %d/%d, 相对变化: %.2e\n', iter, max_iter, relative_change);
    end
end
toc;
% 截断未使用的 change_list
change_list = change_list(1:final_iter);
% --- 以下保留原本的分析与输出逻辑 ---
% 计算峰值信噪比(PSNR)和结构相似性(SSIM)
psnr_denoised = calculatePSNR(original_img, real_image);
psnr_noisy = calculatePSNR(original_img, img_grey);
ssim_denoised = calculateSSIM(original_img, real_image);
ssim_noisy = calculateSSIM(original_img, img_grey);
noise_reduction_percent = (1 - 10^(-(psnr_denoised - psnr_noisy)/10)) * 100;
ssim_improvement = ssim_denoised - ssim_noisy;
% 输出误差信息
fprintf('噪声图像峰值信噪比(PSNR): %.2f dB\n', psnr_noisy);
fprintf('去噪后图像峰值信噪比(PSNR): %.2f dB\n', psnr_denoised);
fprintf('PSNR提升: %.2f dB\n', psnr_denoised - psnr_noisy);
fprintf('噪声减少了: %.2f%%\n', noise_reduction_percent);
fprintf('噪声图像结构相似性(SSIM): %.4f\n', ssim_noisy);
fprintf('去噪后图像结构相似性(SSIM): %.4f\n', ssim_denoised);
fprintf('SSIM提升: %.4f\n', ssim_improvement);
% 显示结果
figure;
subplot(1,3,1); 
imshow(uint8(original_img), []); 
title('原始图像'); 
subplot(1,3,2); 
imshow(uint8(img_grey), []); 
title(sprintf('噪声图像PSNR: %.2f dB, SSIM: %.4f', psnr_noisy, ssim_noisy)); 
subplot(1,3,3); 
imshow(uint8(real_image), []); 
title(sprintf('去噪后图像PSNR: %.2f dB, SSIM: %.4f 噪声减少: %.2f%%', ...
    psnr_denoised, ssim_denoised, noise_reduction_percent)); 
hold off;
figure;
plot(change_list);
title('收敛曲线');

% **--- 新增：保存去噪结果 ---**
% 检查目标文件夹是否存在，不存在则创建
if ~exist(save_folder, 'dir')
    mkdir(save_folder);
    fprintf('创建目标文件夹: %s\n', save_folder);
end

% 将双精度结果转换为 uint8 格式（并限制范围），然后保存
denoised_img_uint8 = uint8(max(min(round(real_image), 255), 0));
imwrite(denoised_img_uint8, full_save_path);
fprintf('\n去噪结果已成功保存到: %s\n', full_save_path);
% **--- 结束新增 ---**


% --- 以下为辅助函数 ---
function img_gray = convertToGray(img)
    if size(img, 3) == 3
        img_gray = rgb2gray(img);
    else
        img_gray = img;
    end
end
% 微调了一下输入参数，不再传 mu, lambda，直接传预计算好的第一项
function f_new = update_f_optimized(numerator_part1, d_xx, d_xy, d_yy, b_xx, b_xy, b_yy, ...
                         H_xx, H_yy, H_xy, denom)
    % 计算分子第二项
    term_xx = (H_xx .* fft2(d_xx - b_xx));
    term_yy = (H_yy .* fft2(d_yy - b_yy));
    term_xy = (2 * H_xy .* fft2(d_xy - b_xy));
    B_he_fft = term_xx + term_yy + term_xy;
    
    % 频域更新
    numerator = numerator_part1 + B_he_fft;
    f_new_fft = numerator ./ denom;
    
    % 进行傅里叶逆变换并取实部简化表达
    f_new = real(ifft2(f_new_fft));
end
function psnr = calculatePSNR(original, processed)
    mse = mean((double(original(:)) - double(processed(:))).^2);
    if mse == 0
        psnr = Inf; 
        return;
    end
    psnr = 10 * log10(255^2 / mse);
end
function ssim_val = calculateSSIM(original, processed)
    K = [0.01 0.03];
    L = 255;
    C1 = (K(1)*L)^2;
    C2 = (K(2)*L)^2;
    window = fspecial('gaussian', 8, 1.5);
    mu1 = filter2(window, original, 'valid');
    mu2 = filter2(window, processed, 'valid');
    mu1_sq = mu1.^2;
    mu2_sq = mu2.^2;
    mu1_mu2 = mu1.*mu2;
    sigma1_sq = filter2(window, original.^2, 'valid') - mu1_sq;
    sigma2_sq = filter2(window, processed.^2, 'valid') - mu2_sq;
    sigma12 = filter2(window, original.*processed, 'valid') - mu1_mu2;
    numerator = (2*mu1_mu2 + C1).*(2*sigma12 + C2);
    denominator = (mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2);
    ssim_val = mean(numerator(:)./denominator(:));
end
% 下面的两个函数共同构造三个微分算子 (完全保留原定义)
function [H_xx, H_yy, H_xy, denom] = setup_frequency_operators(m, n, mu, lambda)
    % 三个二阶微分算子的频域表示
    % 基于标准的空间卷积核
    kernel_xx = [0, 0, 0;
                 1, -2, 1;
                 0, 0, 0];  
    kernel_yy = [0, 1, 0;
                 0, -2, 0;
                 0, 1, 0];
    kernel_xy = [1,-1;
                -1,1] ;
    % 将卷积核扩展到图像尺寸并求其频域表示
    H_xx = create_frequency_response(kernel_xx, m, n);
    H_yy = create_frequency_response(kernel_yy, m, n);
    H_xy = create_frequency_response(kernel_xy, m, n);
    % 对卷积核作位移 (保留原本的循环实现)
    trans_locate1 = zeros(m,n);
    trans_locate2 = zeros(m,n);
    trans_locate3 = zeros(m,n);
    for i = 1 : m
        for j = 1 : n
            trans_locate1(i,j) = exp(-sqrt(-1)*2*pi*(j-1)/n);
        end
    end
    for i = 1 : m
        for j = 1 : n
            trans_locate2(i,j) = exp(-sqrt(-1)*2*pi*(i-1)/m);
        end
    end
    for i = 1 : m
        for j = 1 : n
            trans_locate3(i,j) = exp(-sqrt(-1)*2*pi*((i-1)/m+(j-1)/n));
        end
    end
    H_xx = H_xx .* trans_locate1;
    H_yy = H_yy .* trans_locate2;
    H_xy = H_xy .* trans_locate3;
    % 求更新f需要的分母
    denom = (mu/lambda) + abs(H_xx).^2 + abs(H_yy).^2 + 4 * abs(H_xy).^2;
end
function H = create_frequency_response(kernel, m, n)
    % 从空间卷积核创建频域响应
    [k_m, k_n] = size(kernel);
    kernel_full = zeros(m, n);
    center_m = floor(m/2) + 1;
    center_n = floor(n/2) + 1;
    start_m = center_m - floor(k_m/2);
    start_n = center_n - floor(k_n/2);
    kernel_full(start_m:start_m+k_m-1, start_n:start_n+k_n-1) = kernel;
    kernel_shifted = ifftshift(kernel_full);
    H = fft2(kernel_shifted);
end
function [f_xx, f_yy, f_xy] = compute_gradients(f, H_xx, H_yy, H_xy)
    f_fft = fft2(f);
    f_xx = real(ifft2(H_xx .* f_fft));
    f_yy = real(ifft2(H_yy .* f_fft));
    f_xy = real(ifft2(H_xy .* f_fft));
end
function [d_xx, d_yy, d_xy] = update_d(f_xx, f_yy, f_xy, b_xx, b_yy, b_xy, lambda)
    u_xx = f_xx + b_xx;
    u_yy = f_yy + b_yy;
    u_xy = f_xy + b_xy;
    
    d_xx = shrink(u_xx, 1 / lambda);
    d_yy = shrink(u_yy, 1 / lambda);
    d_xy = shrink(u_xy, 1 / lambda);
end
function [b_xx_new, b_yy_new, b_xy_new] = update_b(f_xx, f_yy, f_xy, ...
                                                  d_xx, d_yy, d_xy, ...
                                                  b_xx, b_yy, b_xy)
    b_xx_new = b_xx + (f_xx - d_xx);
    b_yy_new = b_yy + (f_yy - d_yy);
    b_xy_new = b_xy + (f_xy - d_xy);
end
% 修改版 check_convergence，只返回数值和状态，不再修改数组
function [converged, relative_change] = check_convergence_val(g, g_prev, tolerance, iter)
    if iter == 1
        relative_change = 1.0; % 第一次迭代给一个默认大值
        converged = false;
        return;
    end    
    diff_norm = norm(g(:) - g_prev(:));
    g_norm = norm(g_prev(:));    
    if g_norm > 0
        relative_change = diff_norm / g_norm;
    else
        relative_change = diff_norm;
    end
    converged = (relative_change < tolerance);
end
function y = shrink(x, threshold)
    y = sign(x) .* max(abs(x) - threshold, 0);
end