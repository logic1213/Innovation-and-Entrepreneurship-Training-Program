import numpy as np
from scipy.io import loadmat
from tensorflow.keras.models import load_model
import os

# 模型实际路径
model_path = r'C:\Users\LL\rms_cnn_model.h5'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型文件不存在: {model_path}")

model = load_model(model_path)

def predict_single_rms(file_path, model):
    data = loadmat(file_path)
    rms = data['RMS2D']
    original_shape = rms.shape
    ndim = rms.ndim

    # 目标形状: (样本数, 16, 8, 1)
    target_h, target_w = 16, 8

    # 2维: (16, 8)
    if ndim == 2 and rms.shape == (target_h, target_w):
        sample = rms.reshape(1, target_h, target_w, 1).astype(np.float32)

    # 3维: (n, 16, 8) 或 (16, 8, n)
    elif ndim == 3:
        if rms.shape[1:3] == (target_h, target_w):
            # (n, 16, 8)
            sample = rms.reshape(-1, target_h, target_w, 1).astype(np.float32)
        elif rms.shape[:2] == (target_h, target_w):
            # (16, 8, n)
            sample = np.transpose(rms, (2, 0, 1)).reshape(-1, target_h, target_w, 1).astype(np.float32)
        else:
            raise ValueError(f"无法识别的3D形状: {original_shape}")

    # 4维: (n, 16, 8, 1) 或 (16, 8, 1, n)
    elif ndim == 4:
        if rms.shape[1:3] == (target_h, target_w):
            # (n, 16, 8, c)
            if rms.shape[3] == 1:
                sample = rms.astype(np.float32)
            else:
                # 多通道，取第一个通道
                sample = rms[..., 0:1].astype(np.float32)
        elif rms.shape[:2] == (target_h, target_w):
            # (16, 8, c, n) 或 (16, 8, n, c)
            if rms.shape[2] == 1 and rms.ndim == 4:
                # (16, 8, 1, n)
                sample = np.transpose(rms, (3, 0, 1, 2)).astype(np.float32)
            elif rms.shape[3] == 1:
                # (16, 8, n, 1)
                sample = np.transpose(rms, (2, 0, 1, 3)).astype(np.float32)
            else:
                raise ValueError(f"无法识别的4D形状: {original_shape}")
        else:
            raise ValueError(f"无法识别的4D形状: {original_shape}")

    else:
        raise ValueError(f"不支持的维度 {ndim}: {original_shape}")

    # 归一化（每个样本独立除以最大值）
    for i in range(sample.shape[0]):
        max_val = np.max(sample[i])
        if max_val > 0:
            sample[i] = sample[i] / max_val

    # 预测
    probs = model.predict(sample, verbose=0)
    if probs.shape[0] > 1:
        probs = np.mean(probs, axis=0)   # 多个样本取平均
    pred_class = np.argmax(probs)
    return pred_class, probs

# 测试文件路径（请改为您的实际文件）
test_file = r'D:/test_data/action_sample.mat'  # 如果不存在，自动使用训练数据中的一个文件
if not os.path.exists(test_file):
    print(f"测试文件不存在: {test_file}")
    test_file = r'D:\肌电数据\dataeasy\dataeasy\yu\action8_RMS2D.mat'
    print(f"使用示例文件: {test_file}")

pred_class, probs = predict_single_rms(test_file, model)
print(f"预测动作编号: {pred_class + 1}")
print(f"各类别概率: {probs}")