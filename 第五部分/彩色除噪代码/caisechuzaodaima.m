% =========================================================================
% 完整彩色图像 Hessian-Split Bregman 去噪整合代码
% =========================================================================
clear;
clc;
close all; % 关闭之前的图像窗口

% --- 目标保存路径设置 ---
save_folder = 'C:\Users\HUAWEI\Desktop\数值方法大作业资料包\第五部分\彩色除噪结果图\'; % 目标文件夹
save_filename = '彩色除噪结果1.png';
full_save_path = fullfile(save_folder, save_filename);
% -------------------------------------------------------------------------
% 1. 读取图像与预处理
% -------------------------------------------------------------------------
% 请确保路径正确，建议修改为你的实际图片路径
image_path1 = 'C:\Users\HUAWEI\Desktop\数值方法大作业资料包\第五部分\彩色图像分析图\原始图像\彩色原始图像1.png'; 
try
    img1 = imread(image_path1);
catch
    error('找不到图像文件，请检查路径: %s', image_path1);
end
% 如果是灰度图，转为 RGB 伪彩色以便统一流程
if size(img1, 3) == 1
    img1 = repmat(img1, [1, 1, 3]);
end
original_img = double(img1);
image_path2 = 'C:\Users\HUAWEI\Desktop\数值方法大作业资料包\第五部分\彩色图像分析图\处理图像\彩色处理图像1.png'; 
try
    img2 = imread(image_path2);
catch
    error('找不到图像文件，请检查路径: %s', image_path2);
end
% 如果是灰度图，转为 RGB 伪彩色以便统一流程
if size(img2, 3) == 1
    img2 = repmat(img2, [1, 1, 3]);
end
img_noisy = double(img2);
% 计算初始 PSNR/SSIM (用于后续输出)
% 注意：这里是在去噪前先简单算一下，实际去噪函数里会算更精确的平均值
% -------------------------------------------------------------------------
% 2. 调用封装好的去噪函数
% -------------------------------------------------------------------------
% 算法参数
mu = 185;
lambda = 100;  
max_iter = 5000; % 彩色图计算量大，可适当调整
tolerance = 1e-4;
fprintf('开始去噪处理 (Split Bregman)...\n');
tic;
% 调用下方定义的局部函数
[denoised_img, stats, change_lists] = color_tv_bregman_denoise(img_noisy, original_img, mu, lambda, max_iter, tolerance);
elapsed_time = toc;
fprintf('去噪完成，耗时: %.2f 秒\n', elapsed_time);
% -------------------------------------------------------------------------
% 3. 输出结果 (保留你要求的格式)
% -------------------------------------------------------------------------
% 从 stats 结构体获取平均值，赋值给对应变量名以便打印
psnr_noisy = stats.psnr_noisy_mean;
psnr_denoised = stats.psnr_denoised_mean;
noise_reduction_percent = stats.noise_reduction_percent_mean;
ssim_noisy = stats.ssim_noisy_mean;
ssim_denoised = stats.ssim_denoised_mean;
ssim_improvement = stats.ssim_improvement_mean;
fprintf('\n================ 结果分析 ================\n');
fprintf('噪声图像峰值信噪比(PSNR): %.2f dB\n', psnr_noisy);
fprintf('去噪后图像峰值信噪比(PSNR): %.2f dB\n', psnr_denoised);
fprintf('PSNR提升: %.2f dB\n', psnr_denoised - psnr_noisy);
fprintf('噪声减少了: %.2f%%\n', noise_reduction_percent);
fprintf('噪声图像结构相似性(SSIM): %.4f\n', ssim_noisy);
fprintf('去噪后图像结构相似性(SSIM): %.4f\n', ssim_denoised);
fprintf('SSIM提升: %.4f\n', ssim_improvement);
fprintf('==========================================\n');
% -------------------------------------------------------------------------
% 4. 图像与曲线显示 & 保存结果 (新增保存逻辑)
% -------------------------------------------------------------------------

% 图像显示
figure('Name', '去噪效果对比', 'NumberTitle', 'off');
subplot(1,3,1); 
imshow(uint8(original_img), []); 
title('原始图像'); 
subplot(1,3,2); 
imshow(uint8(img_noisy), []); 
title(sprintf('噪声图像\nPSNR: %.2f dB', psnr_noisy)); 
subplot(1,3,3); 
imshow(uint8(denoised_img), []); 
title(sprintf('去噪图像\nPSNR: %.2f dB (提升 %.2f)', psnr_denoised, psnr_denoised - psnr_noisy)); 

% 曲线显示
figure('Name', '收敛曲线', 'NumberTitle', 'off');
hold on;
colors = {'r', 'g', 'b'};
legends = {'R 通道', 'G 通道', 'B 通道'};
for k = 1:3
    plot(change_lists{k}, 'Color', colors{k}, 'LineWidth', 1.5);
end
legend(legends);
title('RGB 各通道收敛曲线');
xlabel('迭代次数');
ylabel('相对误差');
grid on;
hold off;

% **--- 新增：保存去噪结果 ---**
if ~exist(save_folder, 'dir')
    mkdir(save_folder);
    fprintf('创建目标文件夹: %s\n', save_folder);
end

% denoised_img 是 uint8 格式，直接保存
imwrite(denoised_img, full_save_path);
fprintf('\n去噪结果已成功保存到: %s\n', full_save_path);
% **--- 结束新增 ---**

% =========================================================================
% 以下为局部函数定义 (必须放在脚本文件的最后)
% =========================================================================
function [denoised_img, stats, change_lists] = color_tv_bregman_denoise(noisy_img, original_img, mu, lambda, max_iter, tolerance)
% 封装好的彩色去噪函数
    
    % 数据准备
    noisy_img_d = double(noisy_img);
    if ~isempty(original_img)
        original_img_d = double(original_img);
    else
        original_img_d = [];
    end
    [h, w, ~] = size(noisy_img_d);
    % 预计算频域算子 (只需计算一次，三通道通用)
    [H_xx, H_yy, H_xy, denom] = setup_frequency_operators(h, w, mu, lambda);
    denoised_img_d = zeros(h, w, 3);
    change_lists = cell(3, 1);
    
    % 初始化统计数组
    stats.psnr_noisy = zeros(3,1);
    stats.psnr_denoised = zeros(3,1);
    stats.ssim_noisy = zeros(3,1);
    stats.ssim_denoised = zeros(3,1);
    stats.noise_reduction_percent = zeros(3,1);
    stats.ssim_improvement = zeros(3,1);
    % --- 按通道循环处理 ---
    for c = 1:3
        img_chan = noisy_img_d(:,:,c);
        
        % 初始化变量
        real_image = img_chan;
        real_image_prev = zeros(h, w);
        d_xx = zeros(h, w); d_xy = zeros(h, w); d_yy = zeros(h, w);
        b_xx = zeros(h, w); b_xy = zeros(h, w); b_yy = zeros(h, w);
        
        change_list = zeros(max_iter, 1);
        final_iter = 0;
        
        % 预计算该通道的固定分子项
        v_img_fft = fft2(img_chan);
        numerator_part1 = (mu/lambda) * v_img_fft;
        
        % 迭代优化
        for iter = 1:max_iter
            real_image_prev = real_image;
            
            % 1. 更新 f (频域)
            real_image = update_f_optimized(numerator_part1, d_xx, d_xy, d_yy, ...
                                            b_xx, b_xy, b_yy, H_xx, H_yy, H_xy, denom);
            
            % 2. 计算梯度
            [real_image_xx, real_image_yy, real_image_xy] = compute_gradients(real_image, H_xx, H_yy, H_xy);
            
            % 3. 更新 d (Shrinkage)
            [d_xx, d_yy, d_xy] = update_d(real_image_xx, real_image_yy, real_image_xy, ...
                                          b_xx, b_yy, b_xy, lambda);
            
            % 4. 更新 b (Bregman)
            [b_xx, b_yy, b_xy] = update_b(real_image_xx, real_image_yy, real_image_xy, ...
                                          d_xx, d_yy, d_xy, b_xx, b_yy, b_xy);
            
            % 5. 收敛检测
            [converged, relative_change] = check_convergence_val(real_image, real_image_prev, tolerance, iter);
            change_list(iter) = relative_change;
            final_iter = iter;
            
            if converged
                break;
            end
        end
        
        % 保存该通道结果
        change_lists{c} = change_list(1:final_iter);
        denoised_img_d(:,:,c) = real_image;
        
        % 计算该通道指标
        if ~isempty(original_img_d)
            orig_chan = original_img_d(:,:,c);
            stats.psnr_noisy(c) = calculatePSNR(orig_chan, img_chan);
            stats.psnr_denoised(c) = calculatePSNR(orig_chan, real_image);
            stats.ssim_noisy(c) = calculateSSIM(orig_chan, img_chan);
            stats.ssim_denoised(c) = calculateSSIM(orig_chan, real_image);
            
            noise_red = (1 - 10^(-(stats.psnr_denoised(c) - stats.psnr_noisy(c))/10)) * 100;
            stats.noise_reduction_percent(c) = noise_red;
            stats.ssim_improvement(c) = stats.ssim_denoised(c) - stats.ssim_noisy(c);
        end
    end
    
    % 转换为 uint8 输出
    denoised_img = uint8(min(max(round(denoised_img_d), 0), 255));
    
    % 计算整体平均指标
    if ~isempty(original_img_d)
        stats.psnr_noisy_mean = mean(stats.psnr_noisy);
        stats.psnr_denoised_mean = mean(stats.psnr_denoised);
        stats.ssim_noisy_mean = mean(stats.ssim_noisy);
        stats.ssim_denoised_mean = mean(stats.ssim_denoised);
        stats.noise_reduction_percent_mean = mean(stats.noise_reduction_percent);
        stats.ssim_improvement_mean = mean(stats.ssim_improvement);
    end
end
% --- 辅助计算函数 ---
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
function y = shrink(x, threshold)
    y = sign(x) .* max(abs(x) - threshold, 0);
end
function [f_xx, f_yy, f_xy] = compute_gradients(f, H_xx, H_yy, H_xy)
    f_fft = fft2(f);
    f_xx = real(ifft2(H_xx .* f_fft));
    f_yy = real(ifft2(H_yy .* f_fft));
    f_xy = real(ifft2(H_xy .* f_fft));
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
    disp(relative_change);
end
% --- 算子构造函数 ---
function [H_xx, H_yy, H_xy, denom] = setup_frequency_operators(m, n, mu, lambda)
    kernel_xx = [0, 0, 0; 1, -2, 1; 0, 0, 0];  
    kernel_yy = [0, 1, 0; 0, -2, 0; 0, 1, 0];
    kernel_xy = [1,-1; -1,1];
    H_xx = create_frequency_response(kernel_xx, m, n);
    H_yy = create_frequency_response(kernel_yy, m, n);
    H_xy = create_frequency_response(kernel_xy, m, n);
    
    % 位移算子优化写法 (比双层循环快)
    [J, I] = meshgrid(0:n-1, 0:m-1); % 注意 meshgrid 是 x, y -> col, row
    trans_locate1 = exp(-1i * 2 * pi * J / n);
    trans_locate2 = exp(-1i * 2 * pi * I / m);
    trans_locate3 = exp(-1i * 2 * pi * (I/m + J/n));
    
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
    H = fft2(ifftshift(kernel_full));
end
% --- 评价指标函数 ---
function psnr = calculatePSNR(original, processed)
    mse = mean((double(original(:)) - double(processed(:))).^2);
    if mse == 0
        psnr = Inf; 
    else
        psnr = 10 * log10(255^2 / mse);
    end
end
function ssim_val = calculateSSIM(original, processed)
    % 自定义 SSIM 计算
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