%------------------------------------------------------------------%
% Time-reversal based photoacoustic image reconstruction 
% Author: Mengjie Shi Date: 12-04-2022
% Codes are heavily borrowed from a few k-Wave examples 
% Pre-load:
% PA_raw_data: 1024*128*n (timesteps*numberofchannels*numberofframes)
% sos_map: 384*384 
%------------------------------------------------------------------%

%%
clear all
close all;
%
%% 
load('data_0221_164933.mat');
% 创建计算网格
% assign the grid size and create the computational grid 
Nx = 1536;
Ny = 1536;
dx = 0.025e-3;
dy = 0.025e-3; 
pml_y_size = 64;                % [grid points]
pml_x_size = 96;                % [grid points]
kgrid = kWaveGrid(Nx, dx, Ny, dy);

% 设置重建的时间数组 
% 这段代码设置了重建过程中的时间数组。cfl 是柯朗数，用于确定时间步长的稳定性条件。
% estimated_dt 是估计的时间步长，对应于40 MHz的采样频率。
% t_end 是时间数组的结束时间。kgrid.makeTime 函数根据给定的参数生成时间数组，并将其赋值给 kgrid.t_array。     
% time array for the reconstruction 
cfl=0.3;
estimated_dt = 1/160000000; % corresponds to a sampling frequency of 40 MHz 
t_end = estimated_dt * 8191;
kgrid.makeTime([1400 1600],cfl,t_end);
kgrid.t_array = (0:estimated_dt:t_end);

% remove the initial pressure field from the source structure 
% 从 source 结构体中移除了初始压力场 p0。这是因为在时间反演重建中，初始压力场是未知的。
source.p0=1;    %为了确保 source 结构体中有 p0 字段，然后使用 rmfield 函数将其移除。
source = rmfield(source,'p0');

% US transducers 
% 设置了超声换能器的掩码和方向角。
% sensor.mask 是一个大小为 Nx x Ny 的矩阵，用于定义换能器的位置。
% 循环 for ii = 1:128 将换能器的位置设置为1。
% sensor.directivity_angle 是一个与 sensor.mask 大小相同的矩阵，用于定义换能器的方向角。
sensor.mask = zeros(Nx,Ny);
for ii = 1:128
     sensor.mask(1,(6*ii-5):(6*ii-5+4))=1;
end 
sensor.directivity_angle=zeros(Nx,Ny);
sensor.directivity_angle(sensor.mask == 1) = 0;

% % define source and sensor masks (can be moved out of the outest loop) 
% kerf = 1; 
% groupspacing = 11; 
% element_num = 128; 
% source_shape = reshape((1:groupspacing)' + (0:element_num-1)*(kerf+groupspacing), 1, []);
% x_offset = 1;          % [grid points]
% source_m.u_mask = zeros(Nx, Ny); 
% source_m.u_mask(x_offset, source_shape) = 1  ; 
% % do not definie kerf 
% %source_m.u_mask(x_offset,:)=1;
% sensor_m.mask = source_m.u_mask ;
% sensor_m.directivity_angle = sensor_m.mask;
% sensor_m.directivity_angle(sensor_m.mask==1)=pi/2;
% sensor_m.directivity_size = kgrid.dx;
% 
% source.u_mask=source_m.u_mask;
% sensor=sensor_m;

% assgin recorded PA raw data to the time reveral field
% 分配记录的PA原始数据到时间反转场
num_frame = size(non_filtered_rf_normalized,3);

for k = 1:2 % 对于每一帧数据，首先进行两次上采样，分别是4倍和5倍。然后将上采样后的数据赋值给 sensor.time_reversal_boundary_data。
    % signal upsampling 
    rf_data = non_filtered_rf_normalized(:,:,k);   % 获取 PA_raw_data 的第三维度大小，即帧数 num_frame。
    rfdata_upsampled = resample(rf_data,4,1);   
    rfdata_raw= resample( rfdata_upsampled',5,1);
%     % 首先对 rfdata_upsampled 进行转置，然后对转置后的数据进行上采样
%     rfdata_raw= resample(rfdata_upsampled',5,1);
    sensor.time_reversal_boundary_data = rfdata_raw;

    % sound speed of the propagation medium
    % 根据 sos_map 的大小来设置传播介质的声速 medium.sound_speed。
    % 如果 sos_map 只有一个切片，则直接调整其大小；否则，调整当前帧的切片大小。声速被取整以确保是整数值。
    if size(sos_map_d2,3)==1
        medium.sound_speed = floor(imresize(sos_map_d2,[Nx,Ny]));
    else
        medium.sound_speed = floor(imresize(sos_map_d2(:,:,k),[Nx,Ny])); % sos estimation 
    end
    %medium.sound_speed = 1540; % conventional assumption 
    medium.density=1020;

    % set the input options
    input_args = {'PMLInside', false, 'PMLSize', [pml_x_size, pml_y_size],'PlotPML', false, 'Smooth', false,'PlotLayout',true}; 
    % run the simulation
    p0_recon(:,:,k) = kspaceFirstOrder2DG(kgrid, medium, source, sensor, input_args{:});    
end 

%% plot 
index_frame = 4;
figure,
imagesc(kgrid.y_vec * 1e3, kgrid.x_vec * 1e3, rescale(abs(hilbert(single(p0_recon(:,:,index_frame)))))); 
title('Reconstructed Initial Pressure');
colormap hot
ylabel('x-position [mm]');
xlabel('y-position [mm]');
axis image;
colorbar;
scaleFig(1, 0.8);


