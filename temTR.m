% load('D:\pyw\data_0622_223020.mat.mat'); % 包含 non_filtered_rf_normalized, sos_map 等


% define constants 
f0 = 5e6;       % pulse/transducer centre frequency [Hz]
cycles = 2;       % number of cycles in pulse, usually 2-4
c0 = 1540;      % medium speed of sound [m/s]
rho0 = 1020;    % medium density [kg/m3]
F_number = 2;   % F number for CPWC sequence (i.e. maximum angle)
N = 1;            % number of plane waves in CPWC sequence

% Computational grid
f_max = 7e6; % consider the bandwidth of impulse 
lambda_min = c0/f_max;

% grid size and resolution 
grid_depth=38.4e-3; 
grid_width=38.4e-3;
dx=0.025e-3;% resolution: 25 um  
dz=dx;
Nx=1536;
Nz=1536;
pml_y_size = 96;                % [grid points]
pml_x_size = 96;                % [grid points]
kgrid = kWaveGrid(Nz,dx,Nx,dx); 

% 构建换能器阵列掩码

% medium.density = density_map; 


medium.alpha_coeff = 0.75; 
medium.alpha_power = 1.05; 
medium.BonA=10;


cfl=0.3;
estimated_dt = 1/160000000; 
time_steps  = 8191; % 8*1024-1=8191
t_end = estimated_dt * (time_steps); 
kgrid.makeTime([Nx Nz],cfl,t_end);
kgrid.t_array = (0:estimated_dt:t_end);
% 如果你的空间网格是 [Nx Nz]，例如 Nx=1536; Nz=1536;，那么 makeTime([Nx Nz], cfl, t_end); 最保险。


% 方法1：修改时间步长和时间点数
% 设定时间步长和步数
estimated_dt = 1 / 20e6; % 1/20MHz = 50ns
num_time_points = 2030;   % 设备采集的时间点数
t_end = estimated_dt * (num_time_points - 1); % 仿真总时长

% 生成时间数组
kgrid.t_array = (0:estimated_dt:t_end); % 2030个点

% 方法2：保持原始时间步长（ 1 / 160 MHz），通过下采样将时间点数调整为 2030。

% 传感器
% 定义网格参数
% Nx = 1536; % 网格点数（x 方向）
% Nz = 1536; % 网格点数（z 方向）
% dx = 0.025e-3; % 网格分辨率（25 µm）
radius = 15e-3; % 环形阵列的半径（单位：米）

% 将半径转换为网格点数
radius_grid_points = round(radius / dx);

% 定义传感器的角度范围
angle_min = -310/2 * pi / 180; % 最小角度（弧度）
angle_max = 310/2 * pi / 180;  % 最大角度（弧度）

% 均匀划分有效角度范围
angles = linspace(angle_min, angle_max, 256); % 256 个传感器的角度

% 根据角度计算传感器的坐标
sensor_x = round(Nx/2 + radius_grid_points * cos(angles));
sensor_y = round(Nz/2 + radius_grid_points * sin(angles));

% 更新传感器掩码
sensor_m.mask = zeros(Nx, Nz);
sensor_m.mask(sub2ind([Nx, Nz], sensor_x, sensor_y)) = 1;

% 源与传感器掩膜一致
sensor.mask = sensor_m.mask;
source.u_mask = sensor.mask;

% remove the initial pressure field from the source structure 
% 从 source 结构体中移除了初始压力场 p0。这是因为在时间反演重建中，初始压力场是未知的。
source.p0=1;    %为了确保 source 结构体中有 p0 字段，然后使用 rmfield 函数将其移除。
source = rmfield(source,'p0');



PA_raw_data=non_filtered_rf_normalized;
num_frame = size(PA_raw_data,3);
for k = 1:num_frame % 对于每一帧数据，首先进行两次上采样，分别是4倍和5倍。然后将上采样后的数据赋值给 sensor.time_reversal_boundary_data。
    % signal upsampling 
    rf_data = PA_raw_data(:,:,k);   % 获取 PA_raw_data 的第三维度大小，即帧数 num_frame。
    rfdata_upsampled = resample(double(rf_data),4,1);   
    % rfdata_raw= resample('rfdata_upsampled',5,1);
    % 首先对 rfdata_upsampled 进行转置，然后对转置后的数据进行上采样
    rfdata_raw= resample(rfdata_upsampled',5,1);
    sensor.time_reversal_boundary_data = rfdata_raw;

    % sound speed of the propagation medium
    % 根据 sos_map 的大小来设置传播介质的声速 medium.sound_speed。
    % 如果 sos_map 只有一个切片，则直接调整其大小；否则，调整当前帧的切片大小。声速被取整以确保是整数值。
    if size(sos_map_d2,3)==1
        medium.sound_speed = floor(imresize(sos_map_d2,[Nx,Nz]));
    else
            % 假设 sos_map 原始为 250×250
        sos_map_250 = sos_map_d2(:,:,k);  % 你的原始声速分布
        
        % 1. 计算最外一圈的平均值
        top_row    = sos_map_250(1, :);
        bottom_row = sos_map_250(end, :);
        left_col   = sos_map_250(:, 1);
        right_col  = sos_map_250(:, end);
        
        outer_ring = [top_row, bottom_row, left_col', right_col'];
        % 去掉重复的四个角
        outer_ring = unique(outer_ring);  % 可选，严格去重
        avg_sos = mean(outer_ring);
        
        % 2. 创建 384×384 的大矩阵，并全部赋值为 avg_sos
        sos_map_384 = avg_sos * ones(384, 384);
        
        % 3. 计算中心放置起点
        start_idx = floor((384 - 250)/2) + 1;
        end_idx   = start_idx + 250 - 1;
        
        % 4. 将原 sos_map 放到中心
        sos_map_384(start_idx:end_idx, start_idx:end_idx) = sos_map_250;

        medium.sound_speed = floor(imresize(sos_map_384,[Nx,Nz])); % sos estimation 
    end
    %medium.sound_speed = 1540; % conventional assumption 
    medium.density=1020;

    % set the input options
    input_args = {'Smooth', false, 'PMLInside', false, 'PlotPML', false,'PlotLayout', true,'PMLSize',25};
    
    % run the simulation
    p0_recon(:,:,k) = kspaceFirstOrder2DG(kgrid, medium, source, sensor, input_args{:});    
end 

