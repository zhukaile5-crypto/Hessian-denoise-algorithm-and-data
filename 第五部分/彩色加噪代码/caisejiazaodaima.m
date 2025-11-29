% 读取图像
original_img = imread('C:\Users\HUAWEI\Desktop\原始图像\原始图像10.png');

% 确保是彩色图像，如果是灰度图，则转换为3通道的彩色图 (虽然通常不会这么做，但为了代码健壮性)
if size(original_img, 3) == 1
    original_img = repmat(original_img, [1, 1, 3]);
end

% 设置参数
noise_fraction = 0.5;    % 添加噪声的像素比例 (例如 0.5 表示 50% 的像素会受影响)
sigma = 120;              % 噪声标准差 (对于每个通道的高斯噪声)

% 获取图像信息
[height, width, ~] = size(original_img); % 现在是获取彩色图像的尺寸
total_pixels = height * width;

% 将原始图像转换为双精度，以便进行浮点运算
noisy_img_double = double(original_img);

% 对每个颜色通道独立加噪
num_channels = 3;
for c = 1:num_channels
    % 为当前通道选择随机像素
    num_noise_pixels = round(total_pixels * noise_fraction);
    random_indices = randperm(total_pixels, num_noise_pixels); % 1D 索引，方便操作

    % 生成高斯噪声
    gaussian_noise = sigma * randn(1, num_noise_pixels);

    % 将噪声添加到当前通道的选定像素
    % 注意：这里直接操作二维切片，索引会正确对应到三维数组的某个通道
    current_channel = noisy_img_double(:,:,c);
    current_channel(random_indices) = current_channel(random_indices) + gaussian_noise;
    noisy_img_double(:,:,c) = current_channel;
end

% 限制范围并转换为 uint8
% 每个像素值都应在 [0, 255] 之间
noisy_img_final = uint8(max(min(noisy_img_double, 255), 0));

% 显示结果
figure;
subplot(1, 2, 1);
imshow(original_img);
title('原始彩色图像');

subplot(1, 2, 2);
imshow(noisy_img_final);
title(sprintf('彩色加噪图像 (比例: %.0f%%, Std: %d)', noise_fraction*100, sigma));

% 保存加噪图像
imwrite(noisy_img_final, 'color_noisy.png');