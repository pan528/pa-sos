%% 2D simulation of a single plane wave transmission with kWave and ustb 
%
% author: Mengjie Shi <mengjie.shi@kcl.ac.uk>
% k-Wave: http://www.k-wave.org/
% ustb: https://www.ustb.no/ 
% Codes for ultrasound beamforming with ustb are based on an example of CPWC linear by Alfonso Rodriguez-Molares

%%
clear all
close all;
%

% add ustb path 
% addpath 'C:\Users\msh20local\AppData\Roaming\MathWorks\MATLAB Add-Ons\Toolboxes'
%% ustb 
% define constants 
f0 = 7e6;       % pulse/transducer centre frequency [Hz]
cycles = 2;       % number of cycles in pulse, usually 2-4
c0 = 1540;      % medium speed of sound [m/s]
rho0 = 1020;    % medium density [kg/m3]
F_number = 2;   % F number for CPWC sequence (i.e. maximum angle)
N = 1;            % number of plane waves in CPWC sequence

% define the ultrasound probe as a USTB structure uff.linear_array()
prb = uff.linear_array();
prb.N = 128;                  % number of elements 128
prb.pitch = 300e-6;           % probe pitch in azimuth [m]
prb.element_width = 275e-6;   % element width [m]
prb.element_height = 5000e-6; % element height/transverse length [m]
%fig_handle = prb.plot([],'Linear array');

% define a sequence of plane wave 
seq=uff.wave();
seq.wavefront=uff.wavefront.plane;
seq.apodization = uff.apodization('f_number',F_number,'window',uff.window.hanning,'focus',uff.scan('xyz',[0 0 Inf]));
seq.source.azimuth=0;
%seq.source.distance=-Inf;
seq.probe=prb;
seq.sound_speed=1540;    % reference speed of sound [m/s]
seq.delay = 0;
%seq.source.plot(fig_handle);

%% kWave
% Computational grid
f_max = 1.2*f0; % consider the bandwidth of impulse 
lambda_min = c0/f_max;

% grid size and resolution 
grid_depth=38.4e-3; 
grid_width=38.4e-3;
dx=0.025e-3;% resolution: 25 um  
Nx=1536;
Nz=1536;
pml_y_size = 64;                % [grid points]
pml_x_size = 96;                % [grid points]
kgrid = kWaveGrid(Nz,dx,Nx,dx); 

scan=uff.linear_scan('x_axis', kgrid.x_vec, 'z_axis', kgrid.y_vec-kgrid.y_vec(1));
%

%%
%  pd = makedist('Normal','mu',0.012,'sigma',1);
%  t = truncate(pd,0,0.5);
% 
% %  pd_sos = makedist('Normal');
% %  t_sos = truncate(pd_sos,0.044,0.055);
% 
%  pd_sos = makedist('Uniform',0,1);
%  t_sos = truncate(pd_sos,0.044,0.055);

%% Acoustic Propagation 

total_sim = 500; % total number of simulation

% initialise seeds for generating elliptical inclusions
theta_seed = -60:1:60;
R_x_seed = 80:1:600;
R_y_seed = 80:1:400;
Nx_seed = 100:Nx-100;
Ny_seed = 100:Nz-100;

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
     option=1; % 选择是否设置不同区域的声速分布
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
    for kk=1:num_targets 

        % draw ellipses 
        theta = theta_seed(randi(numel(theta_seed),1));
        R_x = R_x_seed(randi(numel(R_x_seed),1)); 
        R_y = R_y_seed(randi(numel(R_y_seed),1)); 
        C_x = Nx_seed(randi(numel(Nx_seed),1));
        C_y = Ny_seed(randi(numel(Ny_seed),1));
        inclusions_region = makeEllipsoid2D([Nx,Nz],[C_x , C_y ], [R_x, R_y],theta) ;
 
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


medium.density = density_map; 
medium.sound_speed=medium_sos_echo;
medium.sound_speed_ref=min(medium.sound_speed(:));


medium.alpha_coeff = 0.75; 
medium.alpha_power = 1.05; 
medium.BonA=10;


cfl=0.3;
estimated_dt = 1/160000000; 
time_steps  = 8191; % 8*1024-1=8991
t_end = estimated_dt * (time_steps); 
kgrid.makeTime([1400 1600],cfl,t_end);
kgrid.t_array = (0:estimated_dt:t_end);


% define source and sensor masks (can be moved out of the outest loop) 
kerf = 1; 
groupspacing = 11; 
element_num = 128; 
source_shape = reshape((1:groupspacing)' + (0:element_num-1)*(kerf+groupspacing), 1, []);
x_offset = 1;          % [grid points]
source_m.u_mask = zeros(Nx, Nz); 
source_m.u_mask(x_offset, source_shape) = 1  ; 
% do not definie kerf 
%source_m.u_mask(x_offset,:)=1;
sensor_m.mask = source_m.u_mask ;
sensor_m.directivity_angle = sensor_m.mask;
sensor_m.directivity_angle(sensor_m.mask==1)=pi/2;
sensor_m.directivity_size = kgrid.dx;

source.u_mask=source_m.u_mask;
sensor=sensor_m;

%--------------------------------------------------------------------------%
% k-Wave calculation

disp('Launching kWave. This can take a while.');
source_strength = 1e6; %Pa
source.ux = toneBurst(1/kgrid.dt, f0, cycles,'Envelope','Rectangular');   
source.ux = (source_strength ./ (1540*rho0)) .* source.ux;
%source.ux = filterTimeSeries(kgrid, medium,source.ux,'PlotSpectrums',true,'ZeroPhase',true); % low-pass 
source.uy = 0;
source.u_mode ='dirichlet'; %?
sensor.record={'p','p_final'};
input_args = {'PMLInside', false, 'PMLSize', [pml_x_size, pml_y_size],'PlotPML', false, 'Smooth', false}; 
% input_args = {'PMLInside', false, 'PMLSize', [pml_x_size, pml_y_size],'PlotPML', false, 'Smooth', false,'PlotLayout',true}; 
% run the simulation
sensor_data = permute(kspaceFirstOrder2DG(kgrid, medium, source, sensor, input_args{:}),[2 1]);
sensor_data.p(isnan(sensor_data.p))=0;

% assign time series signals to channels accordingly for beamforming 
% Gather element signals
real_sensor_data=sensor_data.p';
%without defining kerf 
groupspacing = 11; % 11 if considering kerf 
dim_size=size(real_sensor_data);
sumnum = groupspacing-1 ; 
element=1;
element_data=zeros(time_steps+1,128); %128
for m = 1:groupspacing:dim_size(2)-sumnum
    element_data(:,element) = (real_sensor_data(:,m)+ real_sensor_data(:,m+1)+ real_sensor_data(:,m+2)+ real_sensor_data(:,m+3)+ real_sensor_data(:,m+4)+ real_sensor_data(:,m+5)+ real_sensor_data(:,m+6)+ real_sensor_data(:,m+7)+real_sensor_data(:,m+8)+real_sensor_data(:,m+9)+real_sensor_data(:,m+10));
    % without defining kerf
    %element_data(:,element) = (real_sensor_data(:,m)+ real_sensor_data(:,m+1)+ real_sensor_data(:,m+2)+ real_sensor_data(:,m+3)+ real_sensor_data(:,m+4)+ real_sensor_data(:,m+5)+ real_sensor_data(:,m+6)+ real_sensor_data(:,m+7)+real_sensor_data(:,m+8)+real_sensor_data(:,m+9)+real_sensor_data(:,m+10)+real_sensor_data(:,m+11)) ;
    element = element+1;
end 


% Time gain compensation 
% create time gain compensation function based on attenuation value and
% round trip distance
element_data(1,:)=0;
t0 = length(source.ux) * kgrid.dt/2;
r = 1540*( (1:length(kgrid.t_array)) * kgrid.dt - t0 ) / 2;   
tone_burst_freq=f0;
tgc_alpha_db_cm = medium.alpha_coeff * (tone_burst_freq * 1e-6)^medium.alpha_power;
tgc_alpha_np_m = tgc_alpha_db_cm / 8.686 * 100;
tgc = exp(tgc_alpha_np_m * 2 * r);
for u=1:prb.N_elements
     TCG_element_data(:,u) = bsxfun(@times, tgc, element_data(:,u)')';
end 

% frequency filtering 
% remove noise by band-pass filtering
filtered_channel_data_BP_wide=tools.band_pass(TCG_element_data,1/kgrid.dt,[3e6 4e6 10e6 11e6]); % 
non_filtered_channel_data=TCG_element_data;
% downsampling to 20 MHz
subsampled_channel_data = resample(non_filtered_channel_data,1,8);
subsampled_channel_data_filtered_BP_wide = resample(filtered_channel_data_BP_wide,1,8);
% zeroing cross-talk artefacts
subsampled_channel_data(1:100,:)=0;
subsampled_channel_data_filtered_BP_wide(1:100,:)=0;

% beamforming with ustb
% create a channel_data handle 
channel_data = uff.channel_data();
channel_data.probe = prb;
channel_data.sequence=seq;
channel_data.initial_time=0;
channel_data.sampling_frequency = 20000000;
channel_data.data = subsampled_channel_data_filtered_BP_wide;
channel_data.data(isnan(channel_data.data))=0;

% save variables for training 
sos_map(:,:,i) = medium_sos;
sos_map_echo(:,:,i)= medium.sound_speed; 
% RF data
raw_data_noTCG(:,:,i)=element_data; % no-filter-no-time-gain-no-subsampled
wide_filtered_rf(:,:,i)=subsampled_channel_data_filtered_BP_wide; 
no_filtered_rf(:,:,i)=subsampled_channel_data;

end 
%% channel-wise normalisation/standardalisation; sos pre-processing 
% RF data
for j = 1:total_sim
    for i = 1:128
       % wide_filtered_rf_normalized(:,i,j)=normalize(wide_filtered_rf(:,i,j));
        non_filtered_rf_normalized(:,i,j)=normalize(no_filtered_rf(:,i,j));
    end 
    sos_map_echo_d2(:,:,j)=imresize(sos_map_echo(:,:,j),[384,384],'bilinear'); 
    sos_map_d2(:,:,j)=imresize(sos_map(:,:,j),[384,384],'bilinear');
end 

%% noise addition 
% quantization noise 
% quantizing a real signal to 64 bits of prescion can be modelled as a
% linear system that adds normally distributed noise with a standard
% deviation of 2^(-p)/sqrt(12) theoretically 
% zero mean 
% std_quantized_noise = 2^(-64)/sqrt(12);
% noise_vals = std_quantized_noise*randn(1024,128)+0;
% non_filtered_rf_normalized = non_filtered_rf_normalized+noise_vals;
%
% add thermal noise to each channel with snr [40 120]dB 
for i =1:total_sim 
    snr_val = 40+80*rand(1);
    no_filtered_rf_norm_noisy(:,:,i)=awgn(non_filtered_rf_normalized(:,:,i),snr_val,'measured');
end 
% add realistic noise 
% index = randi([1 2000],1000,1);
% no_filtered_rf_norm_noisy(1:100,:,index)=noise_normalized(1:100,:,randi([1 256],1000,1));

% 获取当前日期
current_date = datestr(now, 'mmdd_HHMMSS');
% 生成唯一的文件名
filename = sprintf('data_%s.mat', current_date);
% 保存所有数据到同一个 .mat 文件
save(filename, 'non_filtered_rf_normalized', 'sos_map_d2');
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
%% Beamforming with ustb
% scan=uff.linear_scan('x_axis',linspace(scan.x_axis(1),scan.x_axis(end),1536).',...
%     'z_axis',linspace(scan.z_axis(1),scan.z_axis(end),1536).');
% 
% pipe = pipeline();
% pipe.channel_data = channel_data;
% pipe.scan = scan;
% 
% pipe.receive_apodization.window = uff.window.hanning;
% pipe.receive_apodization.f_number = F_number;
% %b=postprocess.();
% %b.dimension = dimension.both;
% das = pipe.go({midprocess.das});
% das.plot([],'DAS'); hold on;

% scan=uff.linear_scan('x_axis', kgrid.x_vec, 'z_axis', kgrid.y_vec-kgrid.y_vec(1));
% scan=uff.linear_scan('x_axis',linspace(scan.x_axis(1),scan.x_axis(end),1536).',...
%     'z_axis',linspace(scan.z_axis(1),scan.z_axis(end),1536).');
% 
% pipe = pipeline();
% pipe.channel_data = channel_data;
% pipe.scan = scan;
% 
% pipe.receive_apodization.window = uff.window.hanning;
% pipe.receive_apodization.f_number = F_number;
% %b=postprocess.();
% %b.dimension = dimension.both;
% das = pipe.go({midprocess.das});
% das.plot([],'DAS'); hold on;

% %% inspect sos and density maps 
% figure;
% subplot(1,2,1);
% imagesc(scan.x_axis*1e3,scan.z_axis*1e3,medium.sound_speed); colormap gray; colorbar; axis equal tight;
% xlabel('x [mm]');
% ylabel('z [mm]');
% 
% title('c_0 [m/s]');
% subplot(1,2,2);
% imagesc(scan.x_axis*1e3,scan.z_axis*1e3,medium.density); colormap gray; colorbar; axis equal tight;
% xlabel('x [mm]');
% ylabel('z [mm]');
% title('\rho [kg/m^3]');







