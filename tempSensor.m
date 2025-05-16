%%
clear all
close all;

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
% % 传感器
% % 定义网格参数
% Nx = 1536; % 网格点数（x 方向）
% Nz = 1536; % 网格点数（z 方向）
% dx = 0.025e-3; % 网格分辨率（25 µm）
% radius = 15e-3; % 环形阵列的半径（单位：米）
% 
% % 将半径转换为网格点数
% radius_grid_points = round(radius / dx);
% 
% % 定义传感器的角度范围
% angle_min = -310/2 * pi / 180; % 最小角度（弧度）
% angle_max = 310/2 * pi / 180;  % 最大角度（弧度）
% % no_sensor_min = -25 * pi / 180; % 无传感器区域的最小角度（弧度）
% % no_sensor_max = 25 * pi / 180;  % 无传感器区域的最大角度（弧度）
% 
% % 均匀划分有效角度范围
% angles = linspace(angle_min, angle_max, 256); % 256 个传感器的角度
% % angles = angles(angles < no_sensor_min | angles > no_sensor_max); % 排除无传感器区域
% 
% % 根据角度计算传感器的坐标
% sensor_x = round(Nx/2 + radius_grid_points * cos(angles));
% sensor_y = round(Nz/2 + radius_grid_points * sin(angles));
% 
% % 更新传感器掩码
% sensor.mask = zeros(Nx, Nz);
% sensor.mask(sub2ind([Nx, Nz], sensor_x, sensor_y)) = 1;
% 
% 
% % 源与传感器掩膜一致
% source.u_mask = sensor.mask;
% % source_m.u_mask 是一个掩膜矩阵（mask matrix），在你的代码中用于定义“声源（source）”的空间分布。具体来说：
% % 它是一个大小为 (Nx, Nz) 的二维矩阵，初始值全为0。
% % 其中值为1的位置，表示在这些点上有源（激励/发射），值为0的位置则没有源。
% % 压力源：在代码中用 p_mask 定义源位置，对应字段如 source.p_mask。
% % 速度源：在代码中用 u_mask 定义源位置，对应字段如 source.u_mask
% % 如果你关心源表面振动（如换能器阵元），建议用速度源。
% % 如果只需模拟点声源或已知的压力激励，可用压力源。
% 
% % 可视化检查传感器分布
% figure;
% imagesc(sensor.mask);
% axis image;
% title('Sensor Distribution');
% xlabel('x (grid points)');
% ylabel('y (grid points)');


%% Acoustic Propagation 

total_sim = 1; % total number of simulation

% initialise seeds for generating elliptical inclusions
theta_seed = -60:1:60;
% R_x_seed = 80:1:600;
% R_y_seed = 80:1:400;
% Nx_seed = 100:Nx-100;
% Ny_seed = 100:Nz-100;

side_x = round(25e-3/dx); % 25mm对应的网格点数
side_z = round(25e-3/dz);

center_x = Nx/2;
center_z = Nz/2;

x_min = round(center_x - side_x/2);
x_max = round(center_x + side_x/2);
z_min = round(center_z - side_z/2);
z_max = round(center_z + side_z/2);

for i = 1:total_sim
    
    % scatters distributions 
    sMatrix = full(sprand(Nx, Nz,0.01));
    scatters_index = find(sMatrix~=0);
    rr=(0.03-(-0.03)).*rand(size(scatters_index,1),1)-0.03; 
    sMatrix(scatters_index)=rr;
    density_map = rho0+rho0.*sMatrix; % density deviation 3%
    
    % speed of sound distributions
    % background
    c0_random_seed = 1400 + (1600-1400)*rand; % uniform distribution in [1400 1600];
    c0_background = c0_random_seed; % sound speed [m/s]
    medium_sos = c0_background *ones(Nz, Nx); 
    medium_sos_echo = c0_background *ones(Nz, Nx);
    % 通过修改 medium_sos 和 medium_sos_echo 的值来实现不同区域的声速分布
     % inclusions 
     option=0; % 选择是否设置不同区域的声速分布
     % 设置不同区域的声速分布-3层
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
    % inclusions 
    num_targets = randi([1,4]); % number of inclusions: 1 - 5
    % 限定椭圆最大半轴
    max_R_x = floor(side_x/2);
    max_R_y = floor(side_z/2);
    
    % 初始化半轴种子
    R_x_seed = 80:1:max_R_x;
    R_y_seed = 80:1:max_R_y;
    for kk = 1:num_targets
        theta = theta_seed(randi(numel(theta_seed),1));
        R_x = R_x_seed(randi(numel(R_x_seed),1));
        R_y = R_y_seed(randi(numel(R_y_seed),1));
        
        % 限制中心位置，确保整个椭圆都在中心区域内
        C_x_min = x_min + R_x;
        C_x_max = x_max - R_x;
        C_y_min = z_min + R_y;
        C_y_max = z_max - R_y;
        % 防止范围倒置
        if C_x_min > C_x_max
            temp = C_x_min; C_x_min = C_x_max; C_x_max = temp;
        end
        if C_y_min > C_y_max
            temp = C_y_min; C_y_min = C_y_max; C_y_max = temp;
        end
        C_x = randi([C_x_min, C_x_max]);
        C_y = randi([C_y_min, C_y_max]);
        
        inclusions_region = makeEllipsoid2D([Nx,Nz],[C_x , C_y],[R_x, R_y],theta);
    
        % speed of sound in inlucisons
        % option 1: being hyperechoic
        sos_val = c0_background+c0_background*(0.02+(0.05)*rand)*1;
        % option 2: total random
        % sos_val = 1400 + (1600-1400)*rand;

    
        % assign speed of sound in inclusions into medium 
        inclu_sos(kk) = sos_val;
        medium_sos(inclusions_region==1)= inclu_sos(kk);
        medium_sos_echo(inclusions_region==1) = inclu_sos(kk);

        % specify echongenicity (hyperechoic) in inclusions by assigning 10% grid points as the scatters 
        index_ellipse = find(inclusions_region==1); 
        hypo_index_ellipse = index_ellipse(randi(numel(index_ellipse),1,round(0.1*numel(index_ellipse))));
        rrr=(0.055-0.044).*rand(size(hypo_index_ellipse,1),1)+0.044;
        medium_sos_echo(hypo_index_ellipse)=inclu_sos(kk)+inclu_sos(kk)*rrr;
        hypo_index_ellipse = [];

         % for simplicity, keep the scatter density in the inclusions same
         % as the background
        density_map(inclusions_region==1)=rho0;
    end
%     for kk=1:num_targets 
% 
%         % draw ellipses 
%         theta = theta_seed(randi(numel(theta_seed),1));
%         R_x = R_x_seed(randi(numel(R_x_seed),1)); 
%         R_y = R_y_seed(randi(numel(R_y_seed),1)); 
%         C_x = Nx_seed(randi(numel(Nx_seed),1));
%         C_y = Ny_seed(randi(numel(Ny_seed),1));
%         inclusions_region = makeEllipsoid2D([Nx,Nz],[C_x , C_y ], [R_x, R_y],theta) ;
%  
%         % speed of sound in inlucisons
%         % option 1: being hyperechoic
%         sos_val = c0_background+c0_background*(0.02+(0.05)*rand)*1;
%         % option 2: total random
%         % sos_val = 1400 + (1600-1400)*rand;
% 
%     
%         % assign speed of sound in inclusions into medium 
%         inclu_sos(kk) = sos_val;
%         medium_sos(inclusions_region==1)= inclu_sos(kk);
%         medium_sos_echo(inclusions_region==1) = inclu_sos(kk);
% 
%         % specify echongenicity (hyperechoic) in inclusions by assigning 10% grid points as the scatters 
%         index_ellipse = find(inclusions_region==1); 
%         hypo_index_ellipse = index_ellipse(randi(numel(index_ellipse),1,round(0.1*numel(index_ellipse))));
%         rrr=(0.055-0.044).*rand(size(hypo_index_ellipse,1),1)+0.044;
%         medium_sos_echo(hypo_index_ellipse)=inclu_sos(kk)+inclu_sos(kk)*rrr;
%         hypo_index_ellipse = [];
% 
%          % for simplicity, keep the scatter density in the inclusions same
%          % as the background
%         density_map(inclusions_region==1)=rho0;
%     end


medium.density = density_map; 
medium.sound_speed=medium_sos_echo;
medium.sound_speed_ref=min(medium.sound_speed(:));


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
% source_m.u_mask 是一个掩膜矩阵（mask matrix），在你的代码中用于定义“声源（source）”的空间分布。具体来说：
% 它是一个大小为 (Nx, Nz) 的二维矩阵，初始值全为0。
% 其中值为1的位置，表示在这些点上有源（激励/发射），值为0的位置则没有源。
% 压力源：在代码中用 p_mask 定义源位置，对应字段如 source.p_mask。
% 速度源：在代码中用 u_mask 定义源位置，对应字段如 source.u_mask
% 如果你关心源表面振动（如换能器阵元），建议用速度源。
% 如果只需模拟点声源或已知的压力激励，可用压力源。

Nx1 = size(medium.density,1);
Nz1 = size(medium.density,2);
x = (0:Nx1-1)*dx*1e3; % 单位mm
z = (0:Nz1-1)*dz*1e3; % 单位mm

figure;
subplot(1,2,1);
imagesc(z, x, medium.density);
xlabel('z [mm]'); ylabel('x [mm]');
colorbar; title('Density Map');
axis equal tight;

subplot(1,2,2);
imagesc(z, x, medium.sound_speed);
xlabel('z [mm]'); ylabel('x [mm]');
colorbar; title('Sound Speed Map');
axis equal tight;

%--------------------------------------------------------------------------%
% % k-Wave calculation
% 
% disp('Launching kWave. This can take a while.');
% source_strength = 1e6; %Pa
% source.ux = toneBurst(1/kgrid.dt, f0, cycles,'Envelope','Rectangular');   
% source.ux = (source_strength ./ (1540*rho0)) .* source.ux;
% %source.ux = filterTimeSeries(kgrid, medium,source.ux,'PlotSpectrums',true,'ZeroPhase',true); % low-pass 
% source.uy = 0;
% source.u_mode ='dirichlet'; %?
% sensor.record={'p','p_final'};
% input_args = {'PMLInside', false, 'PMLSize', [pml_x_size, pml_y_size],'PlotPML', false, 'Smooth', false}; 
% % input_args = {'PMLInside', false, 'PMLSize', [pml_x_size, pml_y_size],'PlotPML', false, 'Smooth', false,'PlotLayout',true}; 
% % run the simulation
% sensor_data = permute(kspaceFirstOrder2DG(kgrid, medium, source, sensor, input_args{:}),[2 1]);
% sensor_data.p(isnan(sensor_data.p))=0;
% 
% % assign time series signals to channels accordingly for beamforming 
% % Gather element signals
% real_sensor_data=sensor_data.p';
% %without defining kerf 
% groupspacing = 11; % 11 if considering kerf 
% dim_size=size(real_sensor_data);
% sumnum = groupspacing-1 ; 
% element=1;
% element_data=zeros(time_steps+1,256); %128
% for m = 1:groupspacing:dim_size(2)-sumnum
%     element_data(:,element) = (real_sensor_data(:,m)+ real_sensor_data(:,m+1)+ real_sensor_data(:,m+2)+ real_sensor_data(:,m+3)+ real_sensor_data(:,m+4)+ real_sensor_data(:,m+5)+ real_sensor_data(:,m+6)+ real_sensor_data(:,m+7)+real_sensor_data(:,m+8)+real_sensor_data(:,m+9)+real_sensor_data(:,m+10));
%     % without defining kerf
%     %element_data(:,element) = (real_sensor_data(:,m)+ real_sensor_data(:,m+1)+ real_sensor_data(:,m+2)+ real_sensor_data(:,m+3)+ real_sensor_data(:,m+4)+ real_sensor_data(:,m+5)+ real_sensor_data(:,m+6)+ real_sensor_data(:,m+7)+real_sensor_data(:,m+8)+real_sensor_data(:,m+9)+real_sensor_data(:,m+10)+real_sensor_data(:,m+11)) ;
%     element = element+1;
% end 
% 
% 
% % Time gain compensation 
% % create time gain compensation function based on attenuation value and
% % round trip distance
% element_data(1,:)=0;
% t0 = length(source.ux) * kgrid.dt/2;
% r = 1540*( (1:length(kgrid.t_array)) * kgrid.dt - t0 ) / 2;   
% tone_burst_freq=f0;
% tgc_alpha_db_cm = medium.alpha_coeff * (tone_burst_freq * 1e-6)^medium.alpha_power;
% tgc_alpha_np_m = tgc_alpha_db_cm / 8.686 * 100;
% tgc = exp(tgc_alpha_np_m * 2 * r);
% % for u=1:prb.N_elements
% %      TCG_element_data(:,u) = bsxfun(@times, tgc, element_data(:,u)')';
% % end 
% 
% % frequency filtering 
% % remove noise by band-pass filtering
% filtered_channel_data_BP_wide=tools.band_pass(TCG_element_data,1/kgrid.dt,[3e6 4e6 10e6 11e6]); % 
% non_filtered_channel_data=TCG_element_data;
% % downsampling to 20 MHz
% subsampled_channel_data = resample(non_filtered_channel_data,1,8);
% subsampled_channel_data_filtered_BP_wide = resample(filtered_channel_data_BP_wide,1,8);
% % zeroing cross-talk artefacts
% subsampled_channel_data(1:100,:)=0;
% subsampled_channel_data_filtered_BP_wide(1:100,:)=0;
% 
% % % beamforming with ustb
% % % create a channel_data handle 
% % channel_data = uff.channel_data();
% % channel_data.probe = prb;
% % channel_data.sequence=seq;
% % channel_data.initial_time=0;
% % channel_data.sampling_frequency = 20000000;
% % channel_data.data = subsampled_channel_data_filtered_BP_wide;
% % channel_data.data(isnan(channel_data.data))=0;
% 
% % save variables for training 
% sos_map(:,:,i) = medium_sos;
% sos_map_echo(:,:,i)= medium.sound_speed; 
% % RF data
% raw_data_noTCG(:,:,i)=element_data; % no-filter-no-time-gain-no-subsampled
% wide_filtered_rf(:,:,i)=subsampled_channel_data_filtered_BP_wide; 
% no_filtered_rf(:,:,i)=subsampled_channel_data;
% 
end 