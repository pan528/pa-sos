function mask = makeEllipsoid2D(grid_size, center, radii, theta)
    % ����һ���յ�����
    mask = zeros(grid_size);
    
    % ��������
    [X, Y] = meshgrid(1:grid_size(2), 1:grid_size(1));
    
    % ����������ת��Ϊ��������ϵ
    X = X - center(1);
    Y = Y - center(2);
    
    % ��ת����ϵ
    X_rot = X * cos(theta) + Y * sin(theta);
    Y_rot = -X * sin(theta) + Y * cos(theta);
    
    % ������Բ������
    mask((X_rot / radii(1)).^2 + (Y_rot / radii(2)).^2 <= 1) = 1;
end
