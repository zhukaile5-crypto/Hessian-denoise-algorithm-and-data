clear;
clc;
% ************************************************************
% 【注意】请将以下文件路径修改为您的实际路径
img1 = imread('C:\Users\HUAWEI\Desktop\数值方法大作业资料包\第一部分\同一图像不同噪声占比\原始图像.png');
img = imread('C:\Users\HUAWEI\Desktop\数值方法大作业资料包\第一部分\同一图像不同噪声占比\噪声占比25%.png');
% ************************************************************

% 转换为灰度-双精图像 (范围 [0, 255] 的 double)
img_grey = convertToGray(img);
original_img = convertToGray(img1);
original_img = double(original_img);
img_grey = double(img_grey);
[m , n] = size(img_grey);

% 参数设置
mu = 100;
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

% 预分配收敛记录数组
change_list = zeros(max_iter, 1);

% 预计算频域算子求解 (只计算一次)
[H_xx, H_yy, H_xy, denom] = setup_frequency_operators(m, n, mu, lambda);

% 预计算图像的FFT以及分子的固定部分
v_img_grey = fft2(img_grey);
numerator_part1 = (mu/lambda) * v_img_grey; 

% 迭代优化
tic; 
final_iter = 0;
for iter = 1:max_iter
    real_image_prev = real_image;
    
    % 更新 f (在频域中求解)
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
    
    % 检查收敛
    [converged, relative_change] = check_convergence_val(real_image, real_image_prev, tolerance, iter);
    
    % 记录误差
    change_list(iter) = relative_change;
    final_iter = iter;
    
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

% --- 计算和输出逻辑 ---
% 由于 original_img 和 real_image 都是 [0, 255] 范围的 double，
% 改进后的 calculatePSNR 函数会使用 255 作为 MAX_I。
psnr_denoised = calculatePSNR(original_img, real_image);
psnr_noisy = calculatePSNR(original_img, img_grey);
ssim_denoised = calculateSSIM(original_img, real_image);
ssim_noisy = calculateSSIM(original_img, img_grey);

psnr_improvement = psnr_denoised - psnr_noisy;
noise_reduction_percent = (1 - 10^(-psnr_improvement/10)) * 100;
ssim_improvement = ssim_denoised - ssim_noisy;

% 输出误差信息
fprintf('\n=== ADMM 图像去噪性能评估 ===\n');
fprintf('噪声图像峰值信噪比(PSNR): %.2f dB\n', psnr_noisy);
fprintf('去噪后图像峰值信噪比(PSNR): %.2f dB\n', psnr_denoised);
fprintf('PSNR提升: %.2f dB\n', psnr_improvement);
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
title(sprintf('噪声图像\nPSNR: %.2f dB, SSIM: %.4f', psnr_noisy, ssim_noisy)); 
subplot(1,3,3); 
imshow(uint8(real_image), []); 
title(sprintf('去噪后图像\nPSNR: %.2f dB, SSIM: %.4f', ...
    psnr_denoised, ssim_denoised)); 
hold off;

figure;
plot(change_list);
title('收敛曲线');

% --- 辅助函数 ---

function img_gray = convertToGray(img)
    if size(img, 3) == 3
        img_gray = rgb2gray(img);
    else
        img_gray = img;
    end
end

% 【新版】鲁棒的 PSNR 计算函数
function psnr = calculatePSNR(original, processed)
    % 确保输入图像是 double 类型用于计算 MSE
    original_d = double(original);
    processed_d = double(processed);

    % --- 确定 MAX_I^2 ---
    % 默认假设是 8 位图像范围
    max_val_sq = 255^2; 

    if isinteger(original)
        % 如果是整数类型 (如 uint8, uint16)，取其数据类型的最大值
        max_val = double(intmax(class(original)));
        max_val_sq = max_val^2;
    elseif isa(original, 'double') || isa(original, 'single')
        % 如果是浮点类型 (double, single)
        % 检查图像是否已经被标准化到 [0, 1] 范围
        if max(original_d(:)) <= 1.001 
            % 如果最大值接近 1，则假定图像已被标准化
            max_val_sq = 1.0;
        else
            % 否则，假定图像处于 [0, 255] 范围
            max_val_sq = 255.0^2; 
        end
    end
    
    % --- 计算 MSE ---
    mse = mean((original_d(:) - processed_d(:)).^2);
    
    % --- 计算 PSNR ---
    if mse == 0
        psnr = Inf; 
        return;
    end
    
    psnr = 10 * log10(max_val_sq / mse);
end

% SSIM 计算函数 (保持不变)
function ssim_val = calculateSSIM(original, processed)
    K = [0.01 0.03];
    L = 255;
    C1 = (K(1)*L)^2;
    C2 = (K(2)*L)^2;
    % 确保输入是 double
    original_d = double(original);
    processed_d = double(processed);
    
    mu1 = filter2(fspecial('gaussian', 8, 1.5), original_d, 'valid');
    mu2 = filter2(fspecial('gaussian', 8, 1.5), processed_d, 'valid');
    sigma1_sq = filter2(fspecial('gaussian', 8, 1.5), original_d.^2, 'valid') - mu1.^2;
    sigma2_sq = filter2(fspecial('gaussian', 8, 1.5), processed_d.^2, 'valid') - mu2.^2;
    sigma12 = filter2(fspecial('gaussian', 8, 1.5), original_d.*processed_d, 'valid') - mu1.*mu2;
    numerator = (2*mu1.*mu2 + C1).*(2*sigma12 + C2);
    denominator = (mu1.^2 + mu2.^2 + C1).*(sigma1_sq + sigma2_sq + C2);
    ssim_val = mean(numerator(:)./denominator(:));
end

function f_new = update_f_optimized(numerator_part1, d_xx, d_xy, d_yy, b_xx, b_xy, b_yy, ...
                         H_xx, H_yy, H_xy, denom)
    term_xx = (H_xx .* fft2(d_xx - b_xx));
    term_yy = (H_yy .* fft2(d_yy - b_yy));
    term_xy = (2 * H_xy .* fft2(d_xy - b_xy));
    B_he_fft = term_xx + term_yy + term_xy;
    
    numerator = numerator_part1 + B_he_fft;
    f_new_fft = numerator ./ denom;
    
    f_new = real(ifft2(f_new_fft));
end

function [H_xx, H_yy, H_xy, denom] = setup_frequency_operators(m, n, mu, lambda)
    kernel_xx = [0, 0, 0; 1, -2, 1; 0, 0, 0];  
    kernel_yy = [0, 1, 0; 0, -2, 0; 0, 1, 0];
    kernel_xy = [1,-1; -1,1];
    
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
            trans_locate2(i,j) = exp(-sqrt(-1)*2*pi*(i-1)/m);
            trans_locate3(i,j) = exp(-sqrt(-1)*2*pi*((i-1)/m+(j-1)/n));
        end
    end
    H_xx = H_xx .* trans_locate1;
    H_yy = H_yy .* trans_locate2;
    H_xy = H_xy .* trans_locate3;
    
    denom = (mu/lambda) + abs(H_xx).^2 + abs(H_yy).^2 + 4 * abs(H_xy).^2;
end

function H = create_frequency_response(kernel, m, n)
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

function [converged, relative_change] = check_convergence_val(g, g_prev, tolerance, iter)
    if iter == 1
        relative_change = 1.0; 
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