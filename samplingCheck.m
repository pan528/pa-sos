%% check
   % ����5��ͼ��
num_images = 5;
num_samples = 4;
total_maps = size(sos_map_d2, 3);

for img_idx = 1:num_images
    % ���ѡ��4������
    random_indices = randperm(total_maps, num_samples);

    % ����һ���µ�ͼ�δ���
    figure;

    % ����4����ͼ
    for i = 1:num_samples
        subplot(2, 2, i); % ����2��2�е���ͼ����
        imagesc(sos_map_d2(:, :, random_indices(i))); % ��������ͼ
        colormap gray; % ������ɫӳ��Ϊ�Ҷ�
        colorbar; % ��ʾ��ɫ��
        axis equal tight; % ���������������Ȳ�������ʾ
    end

    % ��������ͼ�����
    sgtitle('Randomly Selected Speed of Sound Maps');

    % ����ͼ���ļ�
    filename = sprintf('sos_maps_%02d.png', img_idx);
    saveas(gcf, filename);

    % �رյ�ǰͼ�񴰿�
    close(gcf);
end