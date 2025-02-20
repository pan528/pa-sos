# pa-sos
工作日志
## 2-20
1. 下载了训练数据，5.7G，6000张
2. 启用硬盘，整理了一下现有的结果和材料
3. ip地址：10.192.44.228 id:panyuewen pw:123456789pyw@ 连不上
4. git 上传和拉取都不成功，好想放弃orz
5. 重新修改代码，增加了分区调整声速板块
6. 生成数据似乎逐帧保存好一点呢，增加该功能；->已解决，保存一次循环的全部数据，含`no_filtered_rf_norm_noisy` 和 `sos_map_d2` 两个变量，命名格式：背景分层数-散射体材质-数量
7. rf数据增加了agwn，可以避免模型过拟合：
    - 模拟真实环境：在实际的超声成像过程中，RF数据通常会受到各种噪声的影响，包括热噪声、量化噪声和环境噪声。通过在模拟数据中添加AWGN，可以更好地模拟这些噪声，从而使训练数据更接近实际情况。
    - 提高模型的鲁棒性：在训练神经网络时，使用包含噪声的训练数据可以提高模型的鲁棒性，使其在处理实际数据时表现更好。模型将学会在存在噪声的情况下提取有用的特征，从而提高其泛化能力。
    - 防止过拟合：添加噪声可以增加训练数据的多样性，从而防止模型过拟合。过拟合是指模型在训练数据上表现很好，但在测试数据上表现不佳。通过添加噪声，可以使模型更好地适应不同的输入数据，提高其在未见数据上的表现。
    - 增强数据集：添加噪声是一种数据增强技术，可以增加训练数据的数量和多样性，从而提高模型的性能。通过在RF数据中添加不同程度的噪声，可以生成更多的训练样本，帮助模型更好地学习。 
8. 增加检查功能：随机抽取8个sos_map_d2中的声速图并绘制为8个子图
    ```python
    % 随机选择8个索引
    num_samples = 8;
    total_maps = size(sos_map_d2, 3);
    random_indices = randperm(total_maps, num_samples);
    
    % 创建一个新的图形窗口
    figure;
    
    % 绘制8个子图
    for i = 1:num_samples
        subplot(2, 4, i); % 创建2行4列的子图布局
        imagesc(sos_map_d2(:, :, random_indices(i))); % 绘制声速图
        colormap gray; % 设置颜色映射为灰度
        colorbar; % 显示颜色条
        axis equal tight; % 设置坐标轴比例相等并紧凑显示
        title(sprintf('sos map %d', random_indices(i))); % 设置子图标题
    end
    
    % 设置整体图像标题
    sgtitle('Randomly Selected Speed of Sound Maps');
    ```
    进阶版：生成10张包含随机抽取的8个sos_map_d2声速图的图像，并将它们保存到文件中。每张图像将包含8个子图。
```python
   % 生成10张图像
num_images = 10;
num_samples = 8;
total_maps = size(sos_map_d2, 3);

for img_idx = 1:num_images
    % 随机选择8个索引
    random_indices = randperm(total_maps, num_samples);

    % 创建一个新的图形窗口
    figure;

    % 绘制8个子图
    for i = 1:num_samples
        subplot(2, 4, i); % 创建2行4列的子图布局
        imagesc(sos_map_d2(:, :, random_indices(i))); % 绘制声速图
        colormap gray; % 设置颜色映射为灰度
        colorbar; % 显示颜色条
        axis equal tight; % 设置坐标轴比例相等并紧凑显示
        title(sprintf('sos map %d', random_indices(i))); % 设置子图标题
    end

    % 设置整体图像标题
    sgtitle('Randomly Selected Speed of Sound Maps');

    % 保存图像到文件
    filename = sprintf('sos_maps_%02d.png', img_idx);
    saveas(gcf, filename);

    % 关闭当前图像窗口
    close(gcf);
end
```
10. 似乎散射体的大小过大了，位置也不正确，应该更靠下，参考：dx=0.025mm，人体皮肤厚度0.5~4mm ->起始的Nz可选择为20~160;软组织的可以，骨骼的应该小，而且位于下部，且仅有一个，先不做含骨骼的
11. 可以绘制椭圆了！DAS结果还可以：        
    ![das](https://github.com/user-attachments/assets/e143f779-86ac-4e4c-a002-c32448b3e9c5)

## 2-21
- [ ] 周报
- [ ] 模型：
    - [ ] 不同背景声速；
    - [ ] 散射体软组织声速      
              <img width="680" alt="pa参数1" src="https://github.com/user-attachments/assets/fb859154-586c-40f1-912a-acbfb3549e09" />    
               <img width="632" alt="pa参数2" src="https://github.com/user-attachments/assets/72af28d8-01ac-40cb-8272-ede8d734c172" />


