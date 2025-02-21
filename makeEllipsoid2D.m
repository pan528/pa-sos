function mask = makeEllipsoid2D(grid_size, center, radii, theta)
    % 创建一个空的掩码
    mask = zeros(grid_size);
    
    % 生成网格
    [X, Y] = meshgrid(1:grid_size(2), 1:grid_size(1));
    
    % 将网格坐标转换为中心坐标系
    X = X - center(1);
    Y = Y - center(2);
    
    % 旋转坐标系
    X_rot = X * cos(theta) + Y * sin(theta);
    Y_rot = -X * sin(theta) + Y * cos(theta);
    
    % 生成椭圆形掩码
    mask((X_rot / radii(1)).^2 + (Y_rot / radii(2)).^2 <= 1) = 1;
end
