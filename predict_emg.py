"""
predict_emg.py
从原始滤波EMG信号中提取RMS2D特征，并用CNN模型预测手指动作类别
用法: python predict_emg.py --emg_file <emg.mat> --model <model.h5>
"""

import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d
import tensorflow as tf
import argparse
import os
import sys

# ------------------- 参数配置 -------------------
DEFAULT_WIN_LEN = 512
DEFAULT_WIN_INC = 112
N_CHANNELS = 128
IMG_H, IMG_W = 16, 8
NUM_CLASSES = 10

# ------------------- RMS2D 特征提取 -------------------
def compute_rms2d(emg_signal, win_len, win_inc):
    n_samples, n_channels = emg_signal.shape
    if n_channels != N_CHANNELS:
        raise ValueError(f"EMG通道数应为 {N_CHANNELS}，实际为 {n_channels}")

    win_num = (n_samples - win_len) // win_inc + 1
    rms2d_list = []

    for i in range(win_num):
        start = i * win_inc
        end = start + win_len
        segment = emg_signal[start:end, :]
        rms_vec = np.sqrt(np.mean(segment ** 2, axis=0))
        rms_vec = rms_vec.astype(np.float64)
        rms_vec[rms_vec == 0] = np.nan
        rms2d = rms_vec.reshape(IMG_H, IMG_W)

        for col in range(IMG_W):
            col_data = rms2d[:, col]
            if np.all(np.isnan(col_data)):
                continue
            not_nan = ~np.isnan(col_data)
            idx = np.arange(IMG_H)[not_nan]
            values = col_data[not_nan]
            if len(idx) > 1:
                f = interp1d(idx, values, kind='linear', fill_value='extrapolate')
                col_data[:] = f(np.arange(IMG_H))
            elif len(idx) == 1:
                col_data[:] = values[0]
        rms2d_list.append(rms2d)

    return rms2d_list

# ------------------- 预测主函数 -------------------
def predict_from_emg(emg_file, model_path, win_len=DEFAULT_WIN_LEN, win_inc=DEFAULT_WIN_INC):
    # 1. 加载EMG数据，自动识别变量名
    if not os.path.exists(emg_file):
        raise FileNotFoundError(f"EMG文件不存在: {emg_file}")
    mat = loadmat(emg_file)
    
    # 自动查找形状为 (N, 128) 的二维数组
    ignore_keys = ['__header__', '__version__', '__globals__']
    candidates = [k for k in mat.keys() if k not in ignore_keys]
    emg = None
    for key in candidates:
        var = mat[key]
        if isinstance(var, np.ndarray) and var.ndim == 2 and var.shape[1] == N_CHANNELS:
            emg = var
            print(f"自动识别到 EMG 变量: '{key}', 形状 {var.shape}")
            break
    
    if emg is None:
        raise KeyError(f"未找到形状为 (N, {N_CHANNELS}) 的 EMG 数据，文件中包含: {list(mat.keys())}")

    print(f"加载EMG信号: {emg.shape[0]} 个采样点, {emg.shape[1]} 通道")

    # 2. 提取RMS2D特征
    rms2d_list = compute_rms2d(emg, win_len, win_inc)
    if not rms2d_list:
        raise RuntimeError("未生成任何RMS2D窗口，请检查EMG信号长度或窗口参数")
    print(f"生成 {len(rms2d_list)} 个RMS2D样本")

    # 3. 转换为模型输入格式 (n, 16, 8, 1) 并进行归一化
    X = np.array(rms2d_list, dtype=np.float32)
    X = X[..., np.newaxis]
    for i in range(X.shape[0]):
        max_val = np.max(X[i])
        if max_val > 0:
            X[i] = X[i] / max_val

    # 4. 加载模型
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"模型加载成功: {model_path}")

    # 5. 批量预测
    probs = model.predict(X, verbose=0)
    avg_probs = np.mean(probs, axis=0)
    pred_class = np.argmax(avg_probs) + 1
    confidence = avg_probs[pred_class-1]

    return pred_class, confidence, avg_probs

# ------------------- 命令行入口 -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从原始EMG信号预测手指动作")
    parser.add_argument("--emg_file", type=str, required=True,
                        help="滤波后的EMG数据文件 (.mat)，包含变量 'EMG'，形状为 (时间, 128)")
    parser.add_argument("--model", type=str, default="rms_cnn_model.h5",
                        help="训练好的Keras模型路径 (默认: rms_cnn_model.h5)")
    parser.add_argument("--win_len", type=int, default=DEFAULT_WIN_LEN,
                        help=f"滑动窗口长度 (默认: {DEFAULT_WIN_LEN})")
    parser.add_argument("--win_inc", type=int, default=DEFAULT_WIN_INC,
                        help=f"窗口增量 (默认: {DEFAULT_WIN_INC})")
    args = parser.parse_args()

    try:
        pred, conf, all_probs = predict_from_emg(
            args.emg_file, args.model, args.win_len, args.win_inc
        )
        print("\n========== 预测结果 ==========")
        print(f"动作类别: {pred} (1~10)")
        print(f"置信度: {conf:.4f}")
        print("各类别概率:", np.round(all_probs, 4))
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)