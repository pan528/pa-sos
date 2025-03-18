import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import matplotlib
import matplotlib.pyplot as plt
matplotlib.interactive(False)
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import load_model
from datetime import datetime  # 导入 datetime 模块
from model import *
from data_loader import *
from sklearn.model_selection import train_test_split
import glob
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import csv
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
# 使用 Huber 损失和学习率调度器
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

def train(data_input, data_target, input_shape=(128, 1024, 1), dropout=0.5, learning_rate=1e-4, momentum=0.9,
          decay=1e-5, batch_size=10, epochs=150, validation_split=0.2, shuffle=True, model_name='pureUSsimulations', model_flag='pre_train'):

    print("GPU加速状态:", "可用" if tf.config.list_physical_devices('GPU') else "不可用")
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(current_dir, 'models/' + model_name)
    if os.path.exists(save_path) == False:
        os.mkdir(save_path)

    if model_flag == 'pre_train':
        # 相对路径·
        # model_path = r'D:\pyw\learning-based-sos-correction-us-pa-main\saved_models\pre_trained.h5'
        model_path = r"D:\pyw\learning-based-sos-correction-us-pa-main\models\no_pretrained_all\no_pre_train_0318_epoch50_loss30.0234.h5"
        base_model = load_model(model_path)
        base_model.trainable = False
        # model = TF_block(base_model, input_shape=input_shape)
        trainable_layers = ['decoder_conv6', 'decoder_conv7']  # 解冻最后两层
        base_model = configure_base_model(base_model, trainable_layers)
        model = TF_block(base_model, input_shape=input_shape)

    else:
        # base model
        model = decoder_block(encoder_block(input_shape, dropout))

    # optimizer
    # lr scheduler is not used
    # opt = SGD(learning_rate=learning_rate, momentum=0.9, decay=1e-5)  # 旧版本api
    opt = SGD(learning_rate=learning_rate, momentum=0.9, weight_decay=1e-5)
    loss_fn = Huber(delta=1.0)
    # opt = Adam(learning_rate)
    # model.compile(optimizer=opt, loss='mse', metrics=RootMeanSquaredError())    # 旧版本api
    model.compile(optimizer=opt, loss=loss_fn, metrics=RootMeanSquaredError())
    model.summary()
    # 定义学习率调度器
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    # 定义早停回调函数
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # 定义模型保存路径
    checkpoint_path = os.path.join(save_path, 'model_epoch{epoch:02d}_val_loss{val_loss:.4f}.h5')
    # 定义 ModelCheckpoint 回调函数
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,       # 保存模型的路径
        monitor='val_loss',             # 监控验证集损失
        save_best_only=True,            # 仅保存性能最好的模型
        save_weights_only=False,        # 保存整个模型（包括结构和权重）
        mode='min',                     # 当监控指标越小时性能越好
        verbose=1                       # 输出保存信息
    )
    # model training
    # history = model.fit(data_input, data_target, batch_size, epochs, verbose=1, validation_split=validation_split, shuffle=True)
    # 训练模型
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

    # 获取当前日期
    current_date = datetime.now().strftime("%m%d")
    # 获取最后一个 epoch 的损失值
    final_loss = history.history['loss'][-1]

    # model saving
    model.save(save_path + './%s_%s_epoch%d_loss%.4f.h5' % (model_flag, current_date, epochs, final_loss))
    # model.save(save_path+'./%s.h5'%model_flag)
    # show train/valid rMSE loss
    plt.plot(history.history['root_mean_squared_error'])
    plt.plot(history.history['val_root_mean_squared_error'])
    plt.title('rMSE LOSS')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    # plt.ylim([0, 200])
    plt.legend(['train', 'val'], loc='upper right')
    output_filename = os.path.join(save_path, f'{model_name}-rMSE-Loss.png')
    plt.savefig(output_filename)
    plt.close()

if __name__ == '__main__':

    # # load data
    # data_input = extract_signals(r"D:\pyw\3-800\v73-3-800.mat",'non_filtered_rf_normalized')
    # data_target = extract_signals(r"D:\pyw\3-800\v73-3-800.mat",'sos_map_d2')
    # load data from multiple files
    filenames = [r"D:\pyw\1-800\1-800.mat", r"D:\pyw\1-800-2\1-800-2.mat", r"D:\pyw\1-969\v73-org-e-969.mat",r"D:\pyw\2-600\v73-2-600.mat", r"D:\pyw\2-800\v73-2-800.mat", r"D:\pyw\2-800-2\v73-2-800-2.mat", r"D:\pyw\2-800-3\v73-2-800-3.mat", r"D:\pyw\2-800-4\v73-2-800-4.mat", r"D:\pyw\3-500\v73-3-500.mat", r"D:\pyw\3-800\v73-3-800.mat", r"D:\pyw\3-800-2\v73-3-800-2.mat",r"D:\pyw\3-800-3\3-800-3.mat", r"D:\pyw\3-800-4\3-800-4.mat", r"D:\pyw\3-800-5\3-800-5.mat"]
    # filenames = [r"D:\pyw\3-500\v73-3-500.mat",r"D:\pyw\3-800\v73-3-800.mat", r"D:\pyw\3-800-2\v73-3-800-2.mat", r"D:\pyw\3-800-3\data_0226_185749.mat"]
    # filenames = [r"D:\pyw\2-600\v73-2-600.mat", r"D:\pyw\2-800\v73-2-800.mat", r"D:\pyw\2-800-2\v73-2-800-2.mat", r"D:\pyw\2-800-3\v73-2-800-3.mat", r"D:\pyw\2-800-4\v73-2-800-4.mat"]
    data_input = extract_signals_all(filenames, 'non_filtered_rf_normalized')
    data_target = extract_signals_all(filenames, 'sos_map_d2')

     # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(data_input, data_target, test_size=0.2, random_state=42)

    model_name = 'pretrained_all_early_stop'
    train(X_train, y_train, input_shape=(128, 1024, 1), dropout=0.5, learning_rate=1e-4, momentum=0.9,
          decay=1e-5, batch_size=10, epochs=50, validation_split=0.2, shuffle=True, model_name=model_name, model_flag='pre_train')
    # 

    # 查找指定文件夹中的 .h5 文件
    model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models', model_name)
    model_files = glob.glob(os.path.join(model_dir, '*.h5'))
    if not model_files:
        raise FileNotFoundError(f"No .h5 files found in the directory: {model_dir}")

    # 加载找到的 .h5 文件
    trained_model = load_model(model_files[0])
    print(f"Loaded model from {model_files[0]}")

    # 评估模型在测试集上的性能
    test_loss, test_rmse = trained_model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss}, Test RMSE: {test_rmse}')

    # 预测测试集
    y_pred = trained_model.predict(X_test)

    saveDir = r'D:\pyw\learning-based-sos-correction-us-pa-main\predVSgt'  # 指定保存目录
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    # 获取 colorbar 范围
    vmin = min(y_test.min(), y_pred.min())
    vmax = max(y_test.max(), y_pred.max())

    # 可视化预测结果与真实值的比较
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(np.rot90(tf.squeeze(y_pred[i, :, :, :]), k=-1), cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar(fraction=0.047).ax.set_ylabel('SoS (m/s)')
        plt.title(f'Predicted Frame {i + 1}')
    plt.suptitle('Predicted Sound of Speed (SoS) Estimation')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整子图之间的间距
    output_filename = os.path.join(saveDir, f'{model_name}-Predicted-SoS.png')
    plt.savefig(output_filename)
    plt.close()

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(np.rot90(tf.squeeze(y_test[i, :, :, :]), k=-1), cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar(fraction=0.047).ax.set_ylabel('SoS (m/s)')
        plt.title(f'True Frame {i + 1}')
    plt.suptitle('True Sound of Speed (SoS) Estimation')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整子图之间的间距
    output_filename = os.path.join(saveDir, f'{model_name}-True-SoS.png')
    plt.savefig(output_filename)
    plt.close()

    # 计算并可视化差值图像
    diff = y_pred - y_test
    vmin_diff = diff.min()
    vmax_diff = diff.max()
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(np.rot90(tf.squeeze(diff[i, :, :, :]), k=-1), cmap='viridis', vmin=vmin_diff, vmax=vmax_diff)
        plt.colorbar(fraction=0.047).ax.set_ylabel('Difference (SoS)')
        plt.title(f'Difference Frame {i + 1}')
    plt.suptitle('Difference between Predicted and True SoS')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整子图之间的间距
    output_filename = os.path.join(saveDir, f'{model_name}-Difference-SoS.png')
    plt.savefig(output_filename)
    plt.close()


    # 计算量化指标
    mse = mean_squared_error(y_test.flatten(), y_pred.flatten())
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test.flatten(), y_pred.flatten())
    r2 = r2_score(y_test.flatten(), y_pred.flatten())

    # 计算 PSNR 和 SSIM
    psnr_value = psnr(y_test, y_pred, data_range=y_test.max() - y_test.min())
    ssim_value = ssim(y_test, y_pred, data_range=y_test.max() - y_test.min(), win_size=3, channel_axis=-1)

    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    print(f'MAE: {mae}')
    print(f'R^2 Score: {r2}')
    print(f'PSNR: {psnr_value}')
    print(f'SSIM: {ssim_value}')

    # 将量化指标写入 CSV 文件
    csv_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'evaluation_metrics_training.csv')
    file_exists = os.path.isfile(csv_file_path)

    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Model Name', 'MSE', 'RMSE', 'MAE', 'R^2 Score', 'PSNR', 'SSIM'])
        writer.writerow([model_name, mse, rmse, mae, r2, psnr_value, ssim_value])