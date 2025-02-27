import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras.models import load_model
import matplotlib.pyplot as plt
import glob
from data_loader import *
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import csv
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# locate models
model_name = 'pureUSsimulations-test'
current_dir = os.path.dirname(os.path.realpath(__file__))
save_path = os.path.join(current_dir, 'models/' + model_name)


# 查找指定路径中的 .h5 文件
h5_files = glob.glob(os.path.join(save_path, '*.h5'))
if not h5_files:
    raise FileNotFoundError(f"No .h5 files found in the directory: {save_path}")

# 加载找到的 .h5 文件
trained_model = load_model(h5_files[0])
print(f"Loaded model from {h5_files[0]}")

# # 直接指明模型路径
# trained_model = load_model(r'D:\pyw\learning-based-sos-correction-us-pa-main\saved_models\pre_trained.h5')

test_input = extract_signals('./learning-based-sos-correction-us-pa-main/test_data.mat','non_filtered_rf_normalized')
test_output = trained_model.predict(test_input)

# 计算量化指标
true_output = extract_signals('./learning-based-sos-correction-us-pa-main/test_data.mat', 'sos_map_d2')
mse = mean_squared_error(true_output.flatten(), test_output.flatten())
rmse = np.sqrt(mse)
mae = mean_absolute_error(true_output.flatten(), test_output.flatten())
r2 = r2_score(true_output.flatten(), test_output.flatten())

# 计算 PSNR 和 SSIM
psnr_value = psnr(true_output, test_output, data_range=true_output.max() - true_output.min())
ssim_value = ssim(true_output, test_output, data_range=true_output.max() - true_output.min(), win_size=3, channel_axis=-1)

print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'R^2 Score: {r2}')
print(f'PSNR: {psnr_value}')
print(f'SSIM: {ssim_value}')

# 将量化指标写入 CSV 文件
csv_file_path = os.path.join(current_dir, 'evaluation_metrics.csv')
file_exists = os.path.isfile(csv_file_path)

with open(csv_file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(['Model Name', 'MSE', 'RMSE', 'MAE', 'R^2 Score', 'PSNR', 'SSIM'])
    writer.writerow([model_name, mse, rmse, mae, r2, psnr_value, ssim_value])


# 获取 colorbar 范围
vmin = min(true_output.min(), test_output.min())
vmax = max(true_output.max(), test_output.max())

# 可视化预测结果与真实值的比较
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(np.rot90(tf.squeeze(test_output[i, :, :, :]), k=-1), cmap='viridis')
    plt.colorbar(fraction=0.047).ax.set_ylabel('SoS (m/s)')
    plt.title(f'Predicted Frame {i + 1}')
plt.suptitle('Predicted Sound of Speed (SoS) Estimation')
output_filename = os.path.join(save_path, f'{model_name}-Predicted-SoS.png')
plt.savefig(output_filename)
plt.show()

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(np.rot90(tf.squeeze(true_output[i, :, :, :]), k=-1), cmap='viridis')
    plt.colorbar(fraction=0.047).ax.set_ylabel('SoS (m/s)')
    plt.title(f'True Frame {i + 1}')
plt.suptitle('True Sound of Speed (SoS) Estimation')
output_filename = os.path.join(save_path, f'{model_name}-True-SoS.png')
plt.savefig(output_filename)
plt.show()

# 计算并可视化差值图像
diff = test_output - true_output
vmin_diff = diff.min()
vmax_diff = diff.max()
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(np.rot90(tf.squeeze(diff[i, :, :, :]), k=-1), cmap='viridis', vmin=vmin_diff, vmax=vmax_diff)
    plt.colorbar(fraction=0.047).ax.set_ylabel('Difference (SoS)')
    plt.title(f'Difference Frame {i + 1}')
plt.suptitle('Difference between Predicted and True SoS')
output_filename = os.path.join(save_path, f'{model_name}-Difference-SoS.png')
plt.savefig(output_filename)
plt.show()
