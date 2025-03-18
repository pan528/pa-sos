这个任务涉及从 1D 超声波信号预测 2D 声速分布图像，建议使用 **CNN + Transformer + U-Net** 结合的架构。该架构的关键部分包括：

1. **CNN 层**：用于提取时序信号的局部特征。
2. **Transformer 层**：用于捕捉信号的全局依赖关系，提升建模能力。
3. **U-Net 结构**：用于从特征中生成高分辨率的 2D 声速分布图。

### 代码实现
以下是 PyTorch 版本的实现：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.models.resnet import resnet18

# CNN 模块用于初步特征提取
class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=128, out_channels=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.conv(x)  # (B, 256, 1024)

# Transformer 模块用于全局特征建模
class TransformerEncoder(nn.Module):
    def __init__(self, seq_len=1024, embed_dim=256, num_heads=8, depth=4):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        self.encoder_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads),
            num_layers=depth
        )
    
    def forward(self, x):
        x = rearrange(x, 'b c l -> b l c')  # 转换为 (B, 1024, 256)
        x += self.pos_embedding
        x = self.encoder_layers(x)  # (B, 1024, 256)
        return rearrange(x, 'b l c -> b c l')  # 还原形状 (B, 256, 1024)

# U-Net 解码器用于生成 2D 图像
class UNetDecoder(nn.Module):
    def __init__(self, in_channels=256, out_size=(384, 384)):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
        )
        self.out_size = out_size

    def forward(self, x):
        x = self.upsample(x)  # 上采样至接近目标大小
        return F.interpolate(x, size=self.out_size, mode='bilinear', align_corners=False)

# 整体网络
class UltrasoundModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        self.transformer = TransformerEncoder()
        self.unet = UNetDecoder()

    def forward(self, x):
        x = self.cnn(x)  # (B, 256, 1024)
        x = self.transformer(x)  # (B, 256, 1024)
        x = x.unsqueeze(-1)  # (B, 256, 1024, 1)
        x = self.unet(x)  # (B, 1, 384, 384)
        return x

# 测试
model = UltrasoundModel()
input_tensor = torch.randn(4, 128, 1024)  # Batch size = 4
output = model(input_tensor)
print(output.shape)  # 期望输出: (4, 1, 384, 384)
```

### 关键点：
1. **CNN 提取时序特征**，将 128 维通道压缩为 256 维。
2. **Transformer 全局建模**，增强特征表达能力。
3. **U-Net 进行解码**，将 1D 特征映射到 2D 声速分布图像。

这个模型可以用于超声成像任务，比如医学成像、材料无损检测等。你可以用 MSE Loss 或 SSIM Loss 进行训练，目标是让预测的 2D 声速图像与真实数据匹配。


## 基于原来的代码格式实现：
以下是基于你的格式实现的 **CNN + Transformer + U-Net** 结合模型，适用于从超声波信号（1D）预测声速分布图像（2D）。使用 Keras 进行实现。

---

### **Encoder（包含 CNN + Transformer）**
```python
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, Dropout, MaxPool2D, 
                                     UpSampling2D, Concatenate, GaussianNoise, Resizing)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import GlorotUniform
import tensorflow.keras.backend as K

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"), 
            tf.keras.layers.Dense(embed_dim)
        ])
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.norm2(out1 + ffn_output)

def encoder_block(input_shape=(128, 1024, 1), dropout=0.5, transformer_depth=2):
    rf_input = Input(shape=input_shape)
    noisy_rf_input = GaussianNoise(1)(rf_input)

    # CNN 部分
    x = Conv2D(32, (3, 15), (1, 2), padding='same', activation='LeakyReLU', kernel_initializer=GlorotUniform())(noisy_rf_input)
    x = BatchNormalization()(x, training=True)
    x = Conv2D(32, (3, 13), (1, 2), padding='same', activation='LeakyReLU', kernel_initializer=GlorotUniform())(x)
    x = BatchNormalization()(x, training=True)
    x = Conv2D(32, (3, 11), (1, 2), padding='same', activation='LeakyReLU', kernel_initializer=GlorotUniform())(x)
    x = BatchNormalization()(x, training=True)
    x = Conv2D(32, (3, 9), (1, 2), padding='same', activation='LeakyReLU', kernel_initializer=GlorotUniform())(x)
    x = Dropout(dropout)(x)
    x = BatchNormalization()(x, training=True)

    # Transformer 部分
    x_shape = K.int_shape(x)
    x = tf.keras.layers.Reshape((x_shape[1] * x_shape[2], x_shape[3]))(x)
    for _ in range(transformer_depth):
        x = TransformerBlock(embed_dim=x_shape[3], num_heads=8, ff_dim=256)(x)
    x = tf.keras.layers.Reshape((x_shape[1], x_shape[2], x_shape[3]))(x)

    # 编码部分继续
    x = Conv2D(64, (3, 7), (1, 1), padding='same', activation='LeakyReLU', kernel_initializer=GlorotUniform())(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    x = Dropout(dropout)(x)
    x = BatchNormalization()(x, training=True)

    x = Conv2D(128, (3, 5), (1, 1), padding='same', activation='LeakyReLU', kernel_initializer=GlorotUniform())(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    x = BatchNormalization()(x, training=True)

    x = Conv2D(256, (3, 3), (1, 1), padding='same', activation='LeakyReLU', kernel_initializer=GlorotUniform())(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    x = Dropout(dropout)(x)
    x = BatchNormalization()(x, training=True)

    model = Model(rf_input, x)
    model.summary()
    return model
```

---

### **Decoder（U-Net 结构）**
```python
def decoder_block(encoder_block):
    x = Conv2D(128, (3, 3), (1, 1), padding='same', activation='LeakyReLU', kernel_initializer=GlorotUniform())(encoder_block.get_layer(index=-1).output)
    x = UpSampling2D((2, 2), interpolation='bilinear')(x)
    x = BatchNormalization()(x, training=True)
    x = Concatenate()([encoder_block.get_layer(index=-4).output, x])  

    x = Conv2D(64, (3, 5), (1, 1), padding='same', activation='LeakyReLU', kernel_initializer=GlorotUniform())(x)
    x = UpSampling2D((2, 2), interpolation='bilinear')(x)
    x = BatchNormalization()(x, training=True)
    x = Concatenate()([encoder_block.get_layer(index=-7).output, x])  

    x = Conv2D(32, (3, 7), (1, 1), padding='same', activation='LeakyReLU', kernel_initializer=GlorotUniform())(x)
    x = UpSampling2D((2, 2), interpolation='bilinear')(x)
    x = BatchNormalization()(x, training=True)
    x = Concatenate()([encoder_block.get_layer(index=-10).output, x])  

    x = Conv2D(32, (3, 9), (1, 1), padding='same', activation='LeakyReLU', kernel_initializer=GlorotUniform())(x)
    x = UpSampling2D((2, 2), interpolation='bilinear')(x)
    x = BatchNormalization()(x, training=True)

    x = Conv2D(32, (3, 11), (1, 1), padding='same', activation='LeakyReLU', kernel_initializer=GlorotUniform())(x)
    x = Resizing(384, 384)(x)

    x = Conv2D(32, (3, 3), (1, 1), padding='same', activation='LeakyReLU', kernel_initializer=GlorotUniform())(x)
    x = BatchNormalization()(x, training=True)

    x = Conv2D(1, (1, 1), (1, 1), padding='same', activation='sigmoid', kernel_initializer=GlorotUniform())(x)

    model = Model(encoder_block.input, x)
    model.summary()
    return model
```

---

### **构建完整模型**
```python
# 构建 Encoder
encoder_model = encoder_block(input_shape=(128, 1024, 1), dropout=0.5)

# 构建 Decoder
full_model = decoder_block(encoder_model)

# 编译模型
full_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

---

### **总结**
- **CNN** 处理局部时序特征
- **Transformer** 捕捉全局信息
- **U-Net** 进行图像恢复
- **符合你的编码格式**

这样，我们可以高效地将超声波信号映射到 2D 声速分布图。你可以用 `MSE Loss` 进行训练，并通过 `SSIM` 进行评估。🚀
