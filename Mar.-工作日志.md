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

# 3-28 上工
我服了，这玩意。调整版本中
## 调整模型保存格式
![image](https://github.com/user-attachments/assets/24470454-37c1-4631-82f0-15a4c83a362e)          
这个警告是因为您正在使用 HDF5 文件格式（`.h5`）保存模型，而 TensorFlow/Keras 推荐使用新的 `.keras` 文件格式或 TensorFlow SavedModel 格式。以下是解决方法：

---

### **1. 使用 `.keras` 文件格式**
将保存模型的文件扩展名从 `.h5` 改为 `.keras`，以使用新的 Keras 文件格式。

#### 修改代码：
```python
# 修改保存模型的路径
model.save(save_path + './%s_%s_epoch%d_loss%.4f.keras' % (model_flag, current_date, epochs, final_loss))
```

---

### **2. 使用 TensorFlow SavedModel 格式**
如果您不需要特定的文件扩展名，可以直接保存为 TensorFlow 的 SavedModel 格式（默认格式）。

#### 修改代码：
```python
# 保存为 TensorFlow SavedModel 格式
model.save(save_path + './%s_%s_epoch%d_loss%.4f' % (model_flag, current_date, epochs, final_loss))
```
SavedModel 格式会将模型保存为一个目录，而不是单个文件。

---

### **3. 如果需要继续使用 `.h5` 格式**
如果您需要与旧版工具或框架兼容，可以忽略此警告，继续使用 `.h5` 格式保存模型。警告不会影响模型的功能。

---

### **4. 总结**
- **推荐**：切换到 `.keras` 或 TensorFlow SavedModel 格式，以获得更好的兼容性和功能支持。
- **兼容性需求**：如果需要与旧版工具兼容，可以继续使用 `.h5` 格式，但需要接受警告提示。

如果您希望完全消除警告并使用推荐的格式，请选择 **1** 或 **2** 的方法。

## 启用 TensorFlow Profiler分析模型的性能
使用 TensorFlow Profiler 可以帮助您分析模型的性能，包括训练时间、GPU/CPU 使用情况、内存占用等。以下是如何在您的代码中集成 TensorFlow Profiler 的步骤：

---

### **1. 导入 TensorFlow Profiler**
在代码中导入 TensorFlow Profiler：
```python
import tensorflow as tf
```

---

### **2. 启用 TensorFlow Profiler**
在训练模型之前启用 Profiler，并在训练结束后停止 Profiler。

#### 示例代码：
```python
# 定义日志目录
log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')

# 启用 TensorFlow Profiler
tf.profiler.experimental.start(log_dir)

# 训练模型
history = model.fit(
    data_input,
    data_target,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=validation_split,
    shuffle=True,
    callbacks=[lr_scheduler, early_stopping, model_checkpoint]
)

# 停止 TensorFlow Profiler
tf.profiler.experimental.stop()
```

---

### **3. 在 TensorBoard 中查看性能分析**
TensorFlow Profiler 会将性能数据保存到指定的日志目录（如 `logs` 文件夹）。您可以使用 TensorBoard 查看这些数据。

#### 启动 TensorBoard：
在终端中运行以下命令：
```bash
tensorboard --logdir=logs
```

然后打开浏览器，访问 `http://localhost:6006` 查看性能分析结果。

---

### **4. 使用 TensorFlow Profiler 的回调**
您还可以使用 TensorFlow 提供的 `TensorBoardProfilerCallback`，它会自动记录性能数据。

#### 示例代码：
```python
from tensorflow.keras.callbacks import TensorBoard

# 定义日志目录
log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')

# 创建 TensorBoard 回调
tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    profile_batch='10,20'  # 在第 10 到 20 个 batch 中记录性能数据
)

# 训练模型时添加回调
history = model.fit(
    data_input,
    data_target,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=validation_split,
    shuffle=True,
    callbacks=[lr_scheduler, early_stopping, model_checkpoint, tensorboard_callback]
)
```

---

### **5. 分析性能数据**
在 TensorBoard 中，您可以查看以下内容：
- **Overview Page**：显示训练时间、设备利用率等。
- **Trace Viewer**：详细显示每个操作的执行时间。
- **GPU Utilization**：显示 GPU 的使用情况。
- **Memory Profile**：显示内存使用情况。

---

### **6. 示例集成到您的代码**
以下是将 TensorFlow Profiler 集成到您代码中的完整示例：
```python
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

# 定义日志目录
log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')

# 创建 TensorBoard 回调
tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    profile_batch='10,20'  # 在第 10 到 20 个 batch 中记录性能数据
)

# 启用 TensorFlow Profiler
tf.profiler.experimental.start(log_dir)

# 训练模型
history = model.fit(
    data_input,
    data_target,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=validation_split,
    shuffle=True,
    callbacks=[lr_scheduler, early_stopping, model_checkpoint, tensorboard_callback]
)

# 停止 TensorFlow Profiler
tf.profiler.experimental.stop()

print(f"TensorFlow Profiler logs saved to {log_dir}")
```

---

### **7. 注意事项**
- **性能开销**：启用 Profiler 会增加训练时间，建议仅在调试或性能分析时使用。
- **日志存储**：确保日志目录有足够的存储空间，尤其是在长时间训练时。
- **GPU 支持**：如果使用 GPU，确保安装了正确版本的 CUDA 和 cuDNN。

---

### **总结**
- **启用 Profiler**：使用 `tf.profiler.experimental.start()` 和 `tf.profiler.experimental.stop()`。
- **使用 TensorBoard**：通过 `TensorBoard` 回调自动记录性能数据。
- **分析数据**：在 TensorBoard 中查看训练时间、设备利用率和内存使用情况。

通过这些步骤，您可以更好地了解模型的性能瓶颈并优化训练过程。
