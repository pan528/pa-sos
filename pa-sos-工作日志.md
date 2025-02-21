## 2-20
1. 下载了训练数据，5.7G，6000张
2. 启用硬盘，整理了一下现有的结果和材料
3. ip地址：10.192.44.228 id:panyuewen pw:123456789pyw@ 连不上
   - [x] 连上了
5. git 上传和拉取都不成功，好想放弃orz
6. 重新修改代码，增加了分区调整声速板块
7. 生成数据似乎逐帧保存好一点呢，增加该功能；->已解决，保存一次循环的全部数据，含`no_filtered_rf_norm_noisy` 和 `sos_map_d2` 两个变量，命名格式：背景分层数-散射体材质-数量
8. rf数据增加了agwn，可以避免模型过拟合：
    - 模拟真实环境：在实际的超声成像过程中，RF数据通常会受到各种噪声的影响，包括热噪声、量化噪声和环境噪声。通过在模拟数据中添加AWGN，可以更好地模拟这些噪声，从而使训练数据更接近实际情况。
    - 提高模型的鲁棒性：在训练神经网络时，使用包含噪声的训练数据可以提高模型的鲁棒性，使其在处理实际数据时表现更好。模型将学会在存在噪声的情况下提取有用的特征，从而提高其泛化能力。
    - 防止过拟合：添加噪声可以增加训练数据的多样性，从而防止模型过拟合。过拟合是指模型在训练数据上表现很好，但在测试数据上表现不佳。通过添加噪声，可以使模型更好地适应不同的输入数据，提高其在未见数据上的表现。
    - 增强数据集：添加噪声是一种数据增强技术，可以增加训练数据的数量和多样性，从而提高模型的性能。通过在RF数据中添加不同程度的噪声，可以生成更多的训练样本，帮助模型更好地学习。 
9. 增加检查功能：随机抽取8个sos_map_d2中的声速图并绘制为8个子图
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
    进阶版：生成5张包含随机抽取的4个sos_map_d2声速图的图像，并将它们保存到文件中。每张图像将包含4个子图。
```python
%% check
   % 生成5张图像
num_images = 5;
num_samples = 4;
total_maps = size(sos_map_d2, 3);

for img_idx = 1:num_images
    % 随机选择4个索引
    random_indices = randperm(total_maps, num_samples);

    % 创建一个新的图形窗口
    figure;

    % 绘制4个子图
    for i = 1:num_samples
        subplot(2, 2, i); % 创建2行2列的子图布局
        imagesc(sos_map_d2(:, :, random_indices(i))); % 绘制声速图
        colormap gray; % 设置颜色映射为灰度
        colorbar; % 显示颜色条
        axis equal tight; % 设置坐标轴比例相等并紧凑显示
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
10. 似乎散射体的大小过大了，位置也不正确，应该更靠下，参考：dx=0.025mm，人体皮肤厚度0.5-4mm ->起始的Nz可选择为20-160;软组织的可以，骨骼的应该小，而且位于下部，且仅有一个，先不做含骨骼的
11. 可以绘制椭圆了！DAS结果还可以：        
    ![das](https://github.com/user-attachments/assets/e143f779-86ac-4e4c-a002-c32448b3e9c5)

## 2-21
### - [x] 周报
    - [x] 单一背景
    - [x] 2层
    - [x] 3层
### - [x] 模型：
    - [x] 不同背景声速；
    - [x] 散射体软组织声速      
              <img width="680" alt="pa参数1" src="https://github.com/user-attachments/assets/fb859154-586c-40f1-912a-acbfb3549e09" />    
               <img width="632" alt="pa参数2" src="https://github.com/user-attachments/assets/72af28d8-01ac-40cb-8272-ede8d734c172" />

1. 昨晚设置的生成2000张，内存会挤爆写不下，最多到969张，以后还是分批进行比较好
2. 将保存和抽样检查的函数单独模块化；
3. - [x] 1024*128，这对吗？需要改成256吗？看一下是线阵还是环阵
         **是线阵**
4. 增加功能方便查看进程：
```python
   % 输出当前循环次数
    disp(['Running simulation ', num2str(i), ' of ', num2str(total_sim)]);
```
5. 分界面位置随机生成，分界面两边的声速随机在`c_backgroundd * [0.9, 1.1]`之间；
   ```python
   % speed of sound distributions
    % 背景声速分布
    % background
    c0_random_seed = 1400 + (1600-1400)*rand; % uniform distribution in [1400 1600];
    c0_background = c0_random_seed; % sound speed [m/s]
    medium_sos = c0_background *ones(Nz, Nx); 
    medium_sos_echo = c0_background *ones(Nz, Nx);
    % 通过修改 medium_sos 和 medium_sos_echo 的值来实现不同区域的声速分布
    % inclusions 
    % 设置不同区域的声速分布-3层
    option=1 % 选择是否设置不同区域的声速分布
    if option == 1
        % 随机生成分界面位置
        boundary1 = randi([1, Nz/2]);
        boundary2 = randi([boundary1+1, Nz]);

        % 随机生成分界面两边的声速
        sos_upper = c0_background * (0.9 + (1.1 - 0.9) * rand);
        sos_middle = c0_background * (0.9 + (1.1 - 0.9) * rand);
        sos_lower = c0_background * (0.9 + (1.1 - 0.9) * rand);

        % 设置声速分布
        medium_sos(1:boundary1, :) = sos_upper; % 上部区域
        medium_sos(boundary1+1:boundary2, :) = sos_middle; % 中部区域
        medium_sos(boundary2+1:end, :) = sos_lower; % 下部区域

        medium_sos_echo(1:boundary1, :) = sos_upper; % 上部区域
        medium_sos_echo(boundary1+1:boundary2, :) = sos_middle; % 中部区域
        medium_sos_echo(boundary2+1:end, :) = sos_lower; % 下部区域
    end
    % 设置不同区域的声速分布-2层
    if option == 2
        % 随机生成分界面位置
        boundary = randi([1, Nz]);

        % 随机生成分界面两边的声速
        sos_upper = c0_background * (0.9 + (1.1 - 0.9) * rand);
        sos_lower = c0_background * (0.9 + (1.1 - 0.9) * rand);

        % 设置声速分布
        medium_sos(1:boundary, :) = sos_upper; % 上部区域
        medium_sos(boundary+1:end, :) = sos_lower; % 下部区域

        medium_sos_echo(1:boundary, :) = sos_upper; % 上部区域
        medium_sos_echo(boundary+1:end, :) = sos_lower; % 下部区域
    end
   ```
### 传感器
这段代码定义了一个线性阵列传感器（sensor），并设置了其掩码（mask）和方向性（directivity）。具体解释如下：

1. **定义参数**：
    - `kerf = 1`：定义阵元之间的间隙（kerf）。
    - `groupspacing = 11`：定义组间距。
    - `element_num = 128`：定义阵列中的元素数量。

2. **定义源形状**：
    - `source_shape = reshape((1:groupspacing)' + (0:element_num-1)*(kerf+groupspacing), 1, []);`：
        - 生成一个包含源形状的数组，表示每个源元素的位置。

3. **定义源掩码**：
    - `x_offset = 1`：定义源掩码的x方向偏移量。
    - `source_m.u_mask = zeros(Nx, Nz);`：创建一个大小为 `Nx` x `Nz` 的零矩阵，表示源掩码。
    - `source_m.u_mask(x_offset, source_shape) = 1;`：在源掩码矩阵中，将源形状的位置设置为1，表示这些位置是源元素。

4. **定义传感器掩码**：
    - `sensor_m.mask = source_m.u_mask;`：将传感器掩码设置为源掩码。
    - `sensor_m.directivity_angle = sensor_m.mask;`：将传感器的方向性角度设置为传感器掩码。
    - `sensor_m.directivity_angle(sensor_m.mask==1) = pi/2;`：将传感器掩码中值为1的位置的方向性角度设置为 `pi/2`，表示这些位置的传感器方向垂直于阵列平面。
    - `sensor_m.directivity_size = kgrid.dx;`：将传感器的方向性大小设置为 `kgrid.dx`。

5. **将源掩码和传感器掩码赋值给 `source` 和 `sensor` 结构体**：
    - `source.u_mask = source_m.u_mask;`：将源掩码赋值给 `source` 结构体的 `u_mask` 属性。
    - `sensor = sensor_m;`：将传感器掩码赋值给 `sensor` 结构体。

总结：
这段代码定义了一个线性阵列传感器，包含128个元素，每个元素之间有一定的间隙（kerf）。传感器掩码表示传感器的位置和方向性，方向性角度设置为垂直于阵列平面。传感器掩码的大小与计算网格（kgrid）的大小相同。
### 重建函数
目前属于一个没跑通的状态，额，kgrid和mask匹配还有问题:            
错误信息表明在 `kspaceFirstOrder2D` 函数中，尝试将 `sensor.time_reversal_boundary_data` 的数据赋值给 `p(sensor_mask_index)` 时，左侧和右侧的元素数目不同。这通常是由于 `sensor.time_reversal_boundary_data` 的大小与 `sensor.mask` 的大小不匹配。
![image](https://github.com/user-attachments/assets/2d36676e-ad11-41a9-a323-347479481928)

以下是一些可能的解决方案：

1. **检查 `sensor.time_reversal_boundary_data` 的大小**：
    - 确保 `sensor.time_reversal_boundary_data` 的大小与 `sensor.mask` 的大小匹配。

2. **确保 `sensor.mask` 的大小与 `kgrid` 的大小匹配**：
    - 确保 `sensor.mask` 的大小与 `kgrid` 的大小相同。
#### 尝试1：将sensor换成和传播过程完全一样的，其他参数（Nx,Ny,dx,PML参数etc.）也修改一致
![传播过程](https://github.com/user-attachments/assets/a1626168-c04a-44aa-8c97-c6d580fd96a9)
![重建报错](https://github.com/user-attachments/assets/c8966be2-9d41-4c96-8964-b9f53a1f4268)         
但是没什么用kkk      
好奇怪。。。。不过我确实不怎么会用重建工具
#### 解决：上采样！就可以了。。。然后保留原来的mask代码，就是用自己的电脑跑非常地慢
请看vcr，跑一张图需要一个多小时，这就是纯集显的实力：
![image](https://github.com/user-attachments/assets/36b38856-d473-4522-988b-6d82d3cc6499)   
![image](https://github.com/user-attachments/assets/ff1a5f8c-465a-4392-895c-15bae39e0c99)

#### 改进方向：GPU加速✔ 可以跑通，但是，结果很恶心。（用自带的绘图会报错，copilot可以绘图但出来依托）
![image](https://github.com/user-attachments/assets/91e2846f-7309-4b9b-83e4-1e75c55395a7)
![TR初步结果](https://github.com/user-attachments/assets/edd3d4f3-e7bb-41ce-b99c-8eafd42c9d62)


