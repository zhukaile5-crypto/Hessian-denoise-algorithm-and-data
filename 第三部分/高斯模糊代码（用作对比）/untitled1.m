clear;
clc;
% ************************************************************
% 【注意】请将以下文件路径修改为您的实际路径
img1 = imread('C:\Users\HUAWEI\Desktop\数值方法大作业资料包\第一部分\同一图像不同噪声占比\原始图像.png');
img = imread('C:\Users\HUAWEI\Desktop\数值方法大作业资料包\第一部分\同一图像不同噪声占比\噪声占比25%.png');
% ************************************************************

% 彩色转灰度（并保持 uint8 类型用于高斯滤波）
if size(img, 3) == 3
    img_grey = rgb2gray(img);
else
    img_grey = img;
end
if size(img1, 3) == 3
    original_img = rgb2gray(img1);
else
    original_img = img1;
end
% 确保图像尺寸相同（重要！）
if ~isequal(size(original_img), size(img_grey))
    original_img = imresize(original_img, size(img_grey));
    fprintf('注意：原始图像已被重新调整大小以匹配噪声图像\n');
end

% 高斯滤波核
core = [1,2,1;2,4,2;1,2,1]/16;
% 对灰度图像（uint8）进行滤波，使用 double 进行卷积
real_image = conv2(double(img_grey), core, "same");
real_image = uint8(real_image); % 转换回 uint8 用于显示和标准PSNR计算

% 计算性能指标
psnr_denoised = calculatePSNR(original_img, real_image);
psnr_noisy = calculatePSNR(original_img, img_grey);
ssim_denoised = calculateSSIM(original_img, real_image);
ssim_noisy = calculateSSIM(original_img, img_grey);
% 计算改进指标
psnr_improvement = psnr_denoised - psnr_noisy;
noise_reduction_percent = (1 - 10^(-psnr_improvement/10)) * 100;
ssim_improvement = ssim_denoised - ssim_noisy;

% 显示结果对比
figure;
subplot(1,3,1); imshow(original_img); title('原始图像');
subplot(1,3,2); imshow(img_grey); title('噪声图像');
subplot(1,3,3); imshow(real_image); title('去噪后图像');

% 显示结果
fprintf('=== 图像去噪性能评估 (高斯滤波) ===\n');
fprintf('噪声图像峰值信噪比(PSNR): %.2f dB\n', psnr_noisy);
fprintf('去噪后图像峰值信噪比(PSNR): %.2f dB\n', psnr_denoised);
fprintf('PSNR提升: %.2f dB\n', psnr_improvement);
fprintf('噪声减少了: %.2f%%\n', noise_reduction_percent);
fprintf('噪声图像结构相似性(SSIM): %.4f\n', ssim_noisy);
fprintf('去噪后图像结构相似性(SSIM): %.4f\n', ssim_denoised);
fprintf('SSIM提升: %.4f\n', ssim_improvement);

% 性能总结
if psnr_improvement > 0 && ssim_improvement > 0
    fprintf('\n✓ 去噪效果良好，两项指标均有提升\n');
else
    fprintf('\n⚠ 去噪效果需要进一步优化\n');
end

% --- 辅助函数 ---

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