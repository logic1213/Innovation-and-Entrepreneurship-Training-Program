"""
手指动作模式识别：基于RMS2D数据的CNN分类器
根据论文《Decoding Finger movement patterns from microscopic neural drive information based on deep learning》
中的sEMG-RMS对比方法实现，输入为16×8的RMS特征图，输出为10类手指动作。

数据路径：包含三个受试者（dengzhihang, hu, ji），每个受试者有10个动作文件（action1_RMS2D.mat ~ action10_RMS2D.mat）
支持自动识别数据维度：2D (16,8), 3D (n,16,8) 或 (16,8,n), 4D (n,16,8,1) 或 (n,16,8,c)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import tensorflow as tf
from tensorflow.keras import layers, models, utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ====================== 1. 配置参数 ======================
DATA_ROOT = r"D:\肌电数据\dataeasy\dataeasy"   # 数据根目录
SUBJECTS = ['dengzhihang', 'hu', 'ji']        # 受试者文件夹名称
NUM_CLASSES = 10                              # 动作类别数
IMG_HEIGHT, IMG_WIDTH = 16, 8                 # RMS2D特征图尺寸
CHANNEL_MODE = 'first'                        # 多通道处理方式: 'first' 或 'mean'
TEST_SIZE = 0.2                               # 测试集比例
RANDOM_SEED = 42                              # 随机种子
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001

# ====================== 2. 数据加载函数（支持多维度） ======================
def load_rms2d_from_mat(file_path, channel_mode='first'):
    """
    从.mat文件加载RMS2D数据，自动适应2、3、4维数据。
    支持的形状：
        - (16, 8)                     : 单个样本
        - (n_samples, 16, 8)          : 多个样本，无通道维
        - (16, 8, n_samples)          : 多个样本，通道在后
        - (n_samples, 16, 8, 1)       : 多个样本，带通道维（正确格式）
        - (n_samples, 16, 8, c)       : 多个样本，多通道
        - (16, 8, 1, n_samples)       : 四维，通道在前，样本在后
        - (16, 8, n_samples, 1)       : 四维，样本在第三维
    参数:
        file_path: str, .mat文件路径
        channel_mode: str, 多通道处理方式 'first' (取第一通道) 或 'mean' (取平均)
    返回:
        X: numpy数组, 形状 (n_samples, 16, 8, 1)
        n_samples: int
    """
    try:
        data = loadmat(file_path)
        rms_data = data['RMS2D']
    except Exception as e:
        print(f"读取文件失败 {file_path}: {e}")
        return None, None

    original_shape = rms_data.shape
    ndim = rms_data.ndim

    # 目标形状
    target_h, target_w = IMG_HEIGHT, IMG_WIDTH

    # 2维: 单样本
    if ndim == 2:
        if rms_data.shape == (target_h, target_w):
            X = rms_data.reshape(1, target_h, target_w, 1).astype(np.float32)
        else:
            print(f"形状不匹配: {file_path} 的形状为 {original_shape}, 期望 (16,8)")
            return None, None

    # 3维: 多样本无通道
    elif ndim == 3:
        if rms_data.shape[1:3] == (target_h, target_w):
            # (n_samples, 16, 8)
            X = rms_data.reshape(-1, target_h, target_w, 1).astype(np.float32)
        elif rms_data.shape[:2] == (target_h, target_w):
            # (16, 8, n_samples)
            X = np.transpose(rms_data, (2, 0, 1)).reshape(-1, target_h, target_w, 1).astype(np.float32)
        else:
            print(f"形状不匹配: {file_path} 的形状为 {original_shape}, 期望 (?,16,8) 或 (16,8,?)")
            return None, None

    # 4维: 多样本带通道
    elif ndim == 4:
        # 情况1: (n_samples, 16, 8, c)
        if rms_data.shape[1:3] == (target_h, target_w):
            if rms_data.shape[3] == 1:
                X = rms_data.astype(np.float32)
            else:
                # 多通道处理
                if channel_mode == 'first':
                    X = rms_data[..., 0:1].astype(np.float32)
                    print(f"  多通道: 取第一个通道，原通道数 {rms_data.shape[3]}")
                elif channel_mode == 'mean':
                    X = np.mean(rms_data, axis=3, keepdims=True).astype(np.float32)
                    print(f"  多通道: 取平均，原通道数 {rms_data.shape[3]}")
                else:
                    raise ValueError("channel_mode 必须是 'first' 或 'mean'")
        # 情况2: (16, 8, 1, n_samples) 或 (16, 8, n_samples, 1)
        elif rms_data.shape[:2] == (target_h, target_w):
            if rms_data.shape[2] == 1 and rms_data.ndim == 4:
                # (16, 8, 1, n)
                X = np.transpose(rms_data, (3, 0, 1, 2)).astype(np.float32)
            elif rms_data.shape[3] == 1:
                # (16, 8, n, 1)
                X = np.transpose(rms_data, (2, 0, 1, 3)).astype(np.float32)
            else:
                print(f"形状不匹配: {file_path} 的形状为 {original_shape}, 无法处理")
                return None, None
        else:
            print(f"形状不匹配: {file_path} 的形状为 {original_shape}, 期望 (?,16,8,?) 或 (16,8,?,?)")
            return None, None
    else:
        print(f"不支持的数据维度: {file_path} 的维度为 {ndim}, 形状 {original_shape}")
        return None, None

    n_samples = X.shape[0]
    # 确保输出形状为 (n_samples, 16, 8, 1)
    if X.shape[-1] != 1:
        print(f"警告: 输出通道数不为1，当前为 {X.shape[-1]}，取第一个通道")
        X = X[..., 0:1]
    return X, n_samples

# ====================== 3. 构建文件列表并加载数据 ======================
print("正在扫描数据文件...")
X_all = []
y_all = []

for subject in SUBJECTS:
    subject_dir = os.path.join(DATA_ROOT, subject)
    if not os.path.isdir(subject_dir):
        print(f"警告: 目录不存在 {subject_dir}, 跳过")
        continue
    for action_id in range(1, NUM_CLASSES + 1):
        file_name = f"action{action_id}_RMS2D.mat"
        file_path = os.path.join(subject_dir, file_name)
        if not os.path.isfile(file_path):
            print(f"警告: 文件不存在 {file_path}, 跳过")
            continue
        X_data, n_samples = load_rms2d_from_mat(file_path, channel_mode=CHANNEL_MODE)
        if X_data is not None:
            X_all.append(X_data)
            # 标签从0开始：action1 -> 0, action10 -> 9
            y_all.append(np.full(n_samples, action_id - 1, dtype=np.int32))
            print(f"已加载: {file_path} -> 样本数 {n_samples}, 标签 {action_id-1}")

if not X_all:
    raise RuntimeError("未加载到任何数据，请检查数据路径或文件格式！")

# 合并所有数据
X = np.concatenate(X_all, axis=0)
y = np.concatenate(y_all, axis=0)
print(f"\n数据加载完成！总样本数: {X.shape[0]}, 特征图尺寸: {X.shape[1:]}")

# ====================== 4. 数据预处理 ======================
# 归一化：每个样本除以其最大值（避免除以0）
print("正在进行数据归一化...")
for i in range(X.shape[0]):
    max_val = np.max(X[i])
    if max_val > 0:
        X[i] = X[i] / max_val
# 可选：全局标准化，如果数据分布差异较大可启用
# X = (X - np.mean(X)) / np.std(X)

# 划分训练集和测试集（分层采样）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)
print(f"训练集样本数: {X_train.shape[0]}, 测试集样本数: {X_test.shape[0]}")

# 标签转为one-hot编码
y_train_cat = utils.to_categorical(y_train, NUM_CLASSES)
y_test_cat = utils.to_categorical(y_test, NUM_CLASSES)

# ====================== 5. 构建CNN模型 ======================
# 参考论文中的VGG简化结构
model = models.Sequential([
    # Block 1
    layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2), strides=2),
    # Block 2
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    # 展平
    layers.Flatten(),
    # 全连接层
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    # 输出层
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# 编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ====================== 6. 训练模型 ======================
# 早停回调
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
)

# 训练
history = model.fit(
    X_train, y_train_cat,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# ====================== 7. 评估模型 ======================
# 测试集准确率
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n测试集准确率: {test_acc:.4f}")

# 预测测试集
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred_classes)

# 绘制混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f'动作{i+1}' for i in range(NUM_CLASSES)],
            yticklabels=[f'动作{i+1}' for i in range(NUM_CLASSES)])
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.tight_layout()
plt.show()

# 输出分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred_classes,
                            target_names=[f'动作{i+1}' for i in range(NUM_CLASSES)]))

# 绘制训练曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('损失曲线')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.title('准确率曲线')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# ====================== 8. 保存模型 ======================
model.save('rms_cnn_model.h5')
print("模型已保存为 rms_cnn_model.h5")

# ====================== 9. 单样本预测函数（示例） ======================
def predict_single_rms(file_path, model, channel_mode='first'):
    """
    对单个RMS2D数据文件进行预测。
    参数:
        file_path: .mat文件路径，包含变量 'RMS2D'
        model: 训练好的Keras模型
        channel_mode: 多通道处理方式
    返回:
        pred_class: 预测的动作编号（0-9，对应动作1~10）
        probabilities: 每个类别的概率数组
    """
    X_single, n_samples = load_rms2d_from_mat(file_path, channel_mode)
    if X_single is None:
        raise ValueError(f"无法加载文件: {file_path}")
    # 归一化（与训练时一致）
    for i in range(X_single.shape[0]):
        max_val = np.max(X_single[i])
        if max_val > 0:
            X_single[i] = X_single[i] / max_val
    probs = model.predict(X_single, verbose=0)
    if probs.shape[0] == 1:
        probs = probs[0]
    else:
        # 如果文件包含多个样本，返回平均概率
        probs = np.mean(probs, axis=0)
    pred_class = np.argmax(probs)
    return pred_class, probs

# 使用示例（取消注释即可测试）
# test_file = r"D:\肌电数据\dataeasy\dataeasy\dengzhihang\action1_RMS2D.mat"
# pred_class, probs = predict_single_rms(test_file, model)
# print(f"预测动作: {pred_class+1}, 概率分布: {probs}")

print("\n所有流程执行完毕。可使用 predict_single_rms() 函数对新数据进行预测。")