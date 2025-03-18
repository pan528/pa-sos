## 3-3
- [x] 面圣
- [x] 想不出来能干嘛，干点dirty work

## 3-7
- [x] 整理数据dirty-work处理完毕

## 3-9
1. 水周报
2. 环阵：
  ```matlab
  USAGE:
      circle = makeCartCircle(radius, num_points)
      circle = makeCartCircle(radius, num_points, center_pos)
      circle = makeCartCircle(radius, num_points, center_pos, arc_angle)
      circle = makeCartCircle(radius, num_points, center_pos, arc_angle, plot_circle)
 
  INPUTS:
      radius          - circle radius [m]
      num_points      - number of points in the circle
 
  OPTIONAL INPUTS:
      center_pos      - [x, y] position of the circle center [m] 
                        (default = [0, 0])
      arc_angle       - arc angle for incomplete circle [radians]
                        (default = 2*pi)
      plot_circle     - Boolean controlling whether the Cartesian points
                        are plotted (default = false)
 
  OUTPUTS:
      circle          - 2 x num_points array of Cartesian coordinates
  ```
## 3-14
1. 大战tensorflow-gpu，累吐了        
   <img width="428" alt="image" src="https://github.com/user-attachments/assets/bb066a30-d46d-42f9-869f-d0e6f10cd871" />
## 3-16
1. 水了个很有条理的组会报告，以后就按这个思路来
2. 莫名其妙地可以用gpu加速了？？不是很明白。。。

## 3-17
1.增加学习率调度器,使用Huber损失函数：
```python
# 使用 Huber 损失和学习率调度器
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import ReduceLROnPlateau

# 定义优化器和损失函数
opt = SGD(learning_rate=1e-4, momentum=0.9, weight_decay=1e-5)
loss_fn = Huber(delta=1.0)

# 编译模型
model.compile(optimizer=opt, loss=loss_fn, metrics=[RootMeanSquaredError(), 'mae'])

# 定义学习率调度器
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# 训练模型
history = model.fit(data_input, data_target, batch_size=32, epochs=100, validation_split=0.2, shuffle=True, callbacks=[lr_scheduler])
```
2. 将现有所有数据都拿来炼丹了，量特别大，卡又一直有人在用，所以练起来比较慢，1000s/epoch左右，我还忘了注释原来的训练函数，导致其实训练了100个epoch。。。不过到后期其实看得出下降了，以后可以增大epoch数然后加入早停
   ![image](https://github.com/user-attachments/assets/de126b3b-82e9-4b5c-a09e-8866b6767fcc)

## 3-18
1. 引入早停策略：
     ```python
     from tensorflow.keras.callbacks import EarlyStopping
      # 定义早停回调函数
      early_stopping = EarlyStopping(
          monitor='val_loss',  # 监控验证集的损失
          patience=10,         # 如果验证集损失在 10 个 epoch 内没有改善，则停止训练
          restore_best_weights=True  # 恢复验证集损失最小的权重
      )
      
      # 在 callbacks 中添加 early_stopping
      history = model.fit(
          data_input, 
          data_target, 
          batch_size=batch_size, 
          epochs=epochs, 
          verbose=1, 
          validation_split=validation_split, 
          shuffle=True, 
          callbacks=[lr_scheduler, early_stopping]  # 添加早停回调
      )
     ```
2. 保存最优checkpoint：
   ```python
   from tensorflow.keras.callbacks import ModelCheckpoint

# 定义模型保存路径
checkpoint_path = os.path.join(save_path, 'best_model.h5')

# 定义 ModelCheckpoint 回调函数
model_checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,       # 保存模型的路径
    monitor='val_loss',             # 监控的指标（如验证集损失）
    save_best_only=True,            # 仅保存性能最好的模型
    save_weights_only=False,        # 保存整个模型（包括结构和权重）
    mode='min',                     # 当监控指标越小时性能越好
    verbose=1                       # 输出保存信息
)

# 在 callbacks 中添加 model_checkpoint
history = model.fit(
    data_input, 
    data_target, 
    batch_size=batch_size, 
    epochs=epochs, 
    verbose=1, 
    validation_split=validation_split, 
    shuffle=True, 
    callbacks=[lr_scheduler, early_stopping, model_checkpoint]  # 添加 ModelCheckpoint
)
   ```
