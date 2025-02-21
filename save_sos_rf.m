%% channel-wise normalisation/standardalisation; sos pre-processing 
% RF data
for j = 1:969
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
for i =1:969 
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