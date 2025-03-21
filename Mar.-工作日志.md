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
1. 引入早停策略✔：
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
2. 保存最优checkpoint✔(可以在每个 epoch 结束时自动保存模型。即使训练被中断，您仍然可以加载最近保存的模型。)：
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
## 3-19
1. 终于能正常解冻了，呵呵    
       ![image](https://github.com/user-attachments/assets/07000378-5217-49b3-9abc-1afb101aa8dd)
2. 让用户在运行代码时输入 model_name √
3. 开始瞎编loss函数
4. 瞎编了cnn-transformer-unet的大联合网络
5. 炼了一个解冻倒数两层+TF_block的，50epoch，性能没有明显提高，**可以让gpt分析一下原因**
## 3-20
### 1. 关于cnn-transformer-unet的大联合网络：跑不动    
     ![image](https://github.com/user-attachments/assets/5e44fba2-ec1c-4330-bdc2-264b2f25b7a9)
#### 1.1 skip connection进行concatenate时通道数没对齐——>已解决
#### 1.2 如何增加if __name__ == '__main__':测试
#### 1.3 如何检查每一层的张量output shape
#### 1.4 如何打印结构
### 2. unet-transformer
姐们的本质是unet。。。当然文章里没提！**可以把这个当作新的点去给小老板汇报**
#### 2.1 transformer引入位置不同可以做文章，encoder&decoder可以各放一个，看看放在哪里的效果好
#### 2.2 skip connection位置的选择
### 3 batch_size
#### 调小了，太慢（=4）
![image](https://github.com/user-attachments/assets/35d46543-96b7-4b66-b233-19cca7e06845)
#### 调大了 试一下（=16）

## 3-21
### 水周报，交了现有模型和预训练模型在1、2、3层背景下的测试结果图与metrics
### 内存不足问题：
![image](https://github.com/user-attachments/assets/15de8ff9-96f6-4283-888f-b0d16eb8f632)        
错误信息表明，`train_test_split` 函数在尝试分割数据时，无法为一个形状为 `(8695, 384, 384, 1)` 且数据类型为 `float64` 的数组分配约 **9.55 GiB** 的内存。这是由于内存不足导致的。
1. **数据规模过大**：
   - 数据集的大小为 `(8695, 384, 384, 1)`，每个样本占用的内存为 `384 * 384 * 8 bytes = 1.18 MB`（`float64` 类型每个元素占用 8 字节）。
   - 总内存需求为 `8695 * 1.18 MB ≈ 9.55 GiB`，超出了系统可用内存。
2. **数据类型不必要地使用了 `float64`**：
   - `float64` 是双精度浮点数，通常在深度学习中并不需要这么高的精度。
   - 使用 `float32` 或 `float16` 可以显著减少内存占用。
#### **解决办法1：降低精度**✔

将数据从 `float64` 转换为 `float32`，这可以将内存占用减少一半。

在加载数据后，添加以下代码：
```python
data_input = data_input.astype('float32')
data_target = data_target.astype('float32')
```

---

#### **方法 2：分批加载数据**✔
如果数据集过大，可以考虑使用生成器或 `tf.data` API 分批加载数据，而不是一次性加载到内存中。

示例代码：
```python
def data_generator(data_input, data_target, batch_size):
    for i in range(0, len(data_input), batch_size):
        yield data_input[i:i+batch_size], data_target[i:i+batch_size]

# 使用生成器
batch_size = 32
train_gen = data_generator(data_input, data_target, batch_size)
```

---

#### **方法 3：分割数据时使用更小的子集**
如果数据集过大，可以先随机抽取一部分数据进行训练和测试。

示例代码：
```python
subset_size = 2000  # 选择一个较小的子集
data_input = data_input[:subset_size]
data_target = data_target[:subset_size]
```

---
