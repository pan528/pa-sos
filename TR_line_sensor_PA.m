%------------------------------------------------------------------%
% Time-reversal based photoacoustic image reconstruction 
% Author: Mengjie Shi Date: 12-04-2022
% Codes are heavily borrowed from a few k-Wave examples 
% Pre-load:
% PA_raw_data: 1024*128*n (timesteps*numberofchannels*numberofframes)
% sos_map: 384*384 
%------------------------------------------------------------------%
%% 
% ������������
% assign the grid size and create the computational grid 
Nx = 1536;
Ny = 1536;
dx = 0.025e-3;
dy = 0.025e-3; 
kgrid = kWaveGrid(Nx, dx, Ny, dy);

% �����ؽ���ʱ������ 
% ��δ����������ؽ������е�ʱ�����顣cfl �ǿ�����������ȷ��ʱ�䲽�����ȶ���������
% estimated_dt �ǹ��Ƶ�ʱ�䲽������Ӧ��40 MHz�Ĳ���Ƶ�ʡ�
% t_end ��ʱ������Ľ���ʱ�䡣kgrid.makeTime �������ݸ����Ĳ�������ʱ�����飬�����丳ֵ�� kgrid.t_array��     
% time array for the reconstruction 
cfl=0.3;
estimated_dt = 1/160000000; % corresponds to a sampling frequency of 40 MHz 
t_end = estimated_dt * 4095;
kgrid.makeTime([1400 1700],cfl,t_end);
kgrid.t_array = (0:estimated_dt:t_end);

% remove the initial pressure field from the source structure 
% �� source �ṹ�����Ƴ��˳�ʼѹ���� p0��������Ϊ��ʱ�䷴���ؽ��У���ʼѹ������δ֪�ġ�
source.p0=1;    %Ϊ��ȷ�� source �ṹ������ p0 �ֶΣ�Ȼ��ʹ�� rmfield ���������Ƴ���
source = rmfield(source,'p0');

% US transducers 
% �����˳���������������ͷ���ǡ�
% sensor.mask ��һ����СΪ Nx x Ny �ľ������ڶ��廻������λ�á�
% ѭ�� for ii = 1:128 ����������λ������Ϊ1��
% sensor.directivity_angle ��һ���� sensor.mask ��С��ͬ�ľ������ڶ��廻�����ķ���ǡ�
sensor.mask = zeros(Nx,Ny);
for ii = 1:128
     sensor.mask(1,(6*ii-5):(6*ii-5+4))=1;
end 
sensor.directivity_angle=zeros(Nx,Ny);
sensor.directivity_angle(sensor.mask == 1) = 0;

% assgin recorded PA raw data to the time reveral field
% �����¼��PAԭʼ���ݵ�ʱ�䷴ת��
num_frame = size(PA_raw_data,3);


for k = 1:2 % ����ÿһ֡���ݣ����Ƚ��������ϲ������ֱ���4����5����Ȼ���ϲ���������ݸ�ֵ�� sensor.time_reversal_boundary_data��
    % signal upsampling 
    rf_data = PA_raw_data(:,:,k);   % ��ȡ PA_raw_data �ĵ���ά�ȴ�С����֡�� num_frame��
%     rfdata_upsampled = resample(rf_data,4,1);   
%     % rfdata_raw= resample('rfdata_upsampled',5,1);
%     % ���ȶ� rfdata_upsampled ����ת�ã�Ȼ���ת�ú�����ݽ����ϲ���
%     rfdata_raw= resample(rfdata_upsampled',5,1);
    rfdata_raw = rf_data;
    sensor.time_reversal_boundary_data = rfdata_raw;

    % sound speed of the propagation medium
    % ���� sos_map �Ĵ�С�����ô������ʵ����� medium.sound_speed��
    % ��� sos_map ֻ��һ����Ƭ����ֱ�ӵ������С�����򣬵�����ǰ֡����Ƭ��С�����ٱ�ȡ����ȷ��������ֵ��
    if size(sos_map,3)==1
        medium.sound_speed = floor(imresize(sos_map,[Nx,Ny]));
    else
        medium.sound_speed = floor(imresize(sos_map(:,:,k),[Nx,Ny])); % sos estimation 
    end
    %medium.sound_speed = 1540; % conventional assumption 
    medium.density=1020;

    % set the input options
    input_args = {'Smooth', false, 'PMLInside', false, 'PlotPML', false,'PlotLayout', true,'PMLSize',25};
    
    % run the simulation
    p0_recon(:,:,k) = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});    
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


