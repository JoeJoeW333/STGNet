# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# 项目名称: 基于CNN-LSTM的深度学习手势动作识别与优化模型 
# 作者: Joe && 巫海洲
# 版本: 最终版 (2024.12.16)
# 修改记录: 经过9次修改后完成的最终版
# 
# 关键说明:
# 1. 本项目旨在通过深度学习模型对手势数据进行分类，使用了CNN+LSTM的混合模型结构。
# 2. 数据处理部分包括特征提取（均值、标准差、傅里叶变换等）和类别不平衡处理（使用SMOTE）。
# 3. 模型训练过程中使用了EarlyStopping和ModelCheckpoint回调函数，以防止过拟合并保存最佳模型。
# 4. 评估部分包括混淆矩阵、分类报告、准确率曲线和损失曲线，以及每个手势类别的训练和测试准确率。
# 5. 最终版代码优化了输出格式，增加了表格展示和图表样式改进，便于结果分析和可视化。
# 
# 特别感谢:
# - 感谢汪小叶老师的倾情指导，LSTM+CNN的灵感来源于此。
# - 感谢杜俊航同学的热情解答，模型的修改建议来源于他的悉心建议。
# - 感谢开源社区提供的工具和库，如TensorFlow、Scikit-learn、Pandas等。
# 
# 版权声明:
# 本作品为Joe的原创项目，保留所有权利。未经许可，禁止任何形式的复制、修改或分发。
#
# 本人遗憾：
# 1. 由于时间限制，本项目未进行全面的超参数调优，准确率低于我想要的95%。
# 2. 数据集较小，特征提取和处理不到位，影响模型的泛化能力。
# 3. 模型评估指标有待进一步优化。没有让损失函数完全收敛到最优
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, BatchNormalization, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from scipy.fftpack import fft
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from imblearn.over_sampling import SMOTE  # 用于处理类别不平衡问题

# 解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 数据路径
data_path = 'data'

# 定义手势类别
gestures = ['arm_to_left',
            'arm_to_right',
            'close_fist_horizontally',
            'close_fist_perpendicularly',
            'hand_away',
            'hand_closer',
            'hand_down',
            'hand_rotation_palm_down',
            'hand_rotation_palm_up',
            'hand_to_left',
            'hand_to_right',
            'hand_up']

# 读取所有手势数据并提取更多特征
def load_data(data_path, gestures):
    X = []
    y = []
    for gesture_id, gesture in enumerate(gestures):
        gesture_folder = os.path.join(data_path, gesture)
        for file_name in os.listdir(gesture_folder):
            if file_name.endswith('.csv'):
                file_path = os.path.join(gesture_folder, file_name)
                data = pd.read_csv(file_path)

                # 提取特征（例如Range、Velocity、x、y等）
                features = data[['Range', 'Velocity', 'x', 'y']].values
    
                # 提取更多特征，如均值、标准差、最小值、最大值等
                mean_features = np.mean(features, axis=0)
                std_features = np.std(features, axis=0)
                min_features = np.min(features, axis=0)
                max_features = np.max(features, axis=0)
    
                # 计算时序特征：例如变化率、傅里叶变换等
                diff_features = np.diff(features, axis=0)
                mean_diff = np.mean(diff_features, axis=0)
                std_diff = np.std(diff_features, axis=0)
    
                # 添加傅里叶变换特征
                fft_features = np.abs(fft(features, axis=0))
                mean_fft = np.mean(fft_features, axis=0)
                std_fft = np.std(fft_features, axis=0)
    
                # 将所有特征组合
                combined_features = np.concatenate([mean_features, std_features, min_features, max_features,
                                                    mean_diff, std_diff, mean_fft, std_fft])
    
                X.append(combined_features)
                y.append(gesture_id)
    
    return np.array(X), np.array(y)

# 加载数据
X, y = load_data(data_path, gestures)

# 处理类别不平衡问题（使用SMOTE）
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 将标签转换为one-hot编码
y_train = to_categorical(y_train, num_classes=len(gestures))
y_test = to_categorical(y_test, num_classes=len(gestures))

# 构建优化后的CNN+LSTM模型
model = Sequential()
model.add(Input(shape=(X_train.shape[1], 1)))  # 使用 Input 对象作为第一层
model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(len(gestures), activation='softmax'))

# 编译模型
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# 设置ModelCheckpoint回调，保存验证集准确率最高的模型
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# 设置EarlyStopping回调，监控val_accuracy和val_loss，连续2000个epoch没有提升则停止训练
early_stopping = EarlyStopping(monitor='val_accuracy', patience=2000, restore_best_weights=True, mode='max', verbose=1)

# 训练模型
history = model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train,
                    epochs=1000, batch_size=128, validation_split=0.2, callbacks=[checkpoint, early_stopping])

# 加载验证集准确率最高的模型
model.load_weights('best_model.keras')

# 评估模型
y_pred = model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1))
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# 计算准确率
accuracy = accuracy_score(y_true, y_pred_classes)
print(f"模型整体准确率: {accuracy * 100:.2f}%")

# 输出分类报告
print("分类报告:")
print(classification_report(y_true, y_pred_classes, target_names=gestures))

# 绘制混淆矩阵
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=gestures, yticklabels=gestures)
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.show()

# 绘制准确率曲线
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.xlabel('Epoch')
plt.ylabel('准确率')
plt.title('训练和验证准确率曲线')
plt.legend()
plt.show()

# 输出每个手势的训练准确率和测试准确率
train_pred = model.predict(X_train.reshape(X_train.shape[0], X_train.shape[1], 1))
train_pred_classes = np.argmax(train_pred, axis=1)
train_true = np.argmax(y_train, axis=1)

train_cm = confusion_matrix(train_true, train_pred_classes)
train_class_accuracy = train_cm.diagonal() / train_cm.sum(axis=1)

test_class_accuracy = cm.diagonal() / cm.sum(axis=1)

# 创建表格数据
data = {
    'Gesture': gestures,
    'Training Accuracy': [f"{acc * 100:.2f}%" for acc in train_class_accuracy],
    'Testing Accuracy': [f"{acc * 100:.2f}%" for acc in test_class_accuracy]
}

df = pd.DataFrame(data)

# 打印表格
print(df.to_string(index=False))

# 绘制每个类别的准确率
fig, ax = plt.subplots(figsize=(12, 6))
width = 0.4

rects1 = ax.bar(df['Gesture'], [float(acc.strip('%')) for acc in df['Training Accuracy']], width, label='Training Accuracy')
rects2 = ax.bar([x + width for x in range(len(df['Gesture']))], [float(acc.strip('%')) for acc in df['Testing Accuracy']], width, label='Testing Accuracy')

ax.set_ylabel('Accuracy (%)')
ax.set_xlabel('Gesture')
ax.set_title('Training and Testing Accuracy for Each Gesture')
ax.set_xticklabels(df['Gesture'], rotation=45, ha="right")
ax.legend()

# 添加水平虚线
ax.axhline(y=90, color='r', linestyle='--', label='90% Accuracy Threshold')

# 添加标签
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        if height == int(height):
            label = f'{int(height)}'
        else:
            label = f'{height:.2f}'
        ax.annotate(label,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8)

autolabel(rects1)
autolabel(rects2)

ax.legend()
fig.tight_layout()
plt.show()

# 选择满足条件的模型版本
best_model_version = None
best_accuracy = 0

for epoch in range(len(history.history['val_accuracy'])):
    if all(acc >= 0.95 for acc in test_class_accuracy):
        if history.history['val_accuracy'][epoch] > best_accuracy:
            best_accuracy = history.history['val_accuracy'][epoch]
            best_model_version = epoch

if best_model_version is not None:
    print(f"选择第 {best_model_version} 个epoch的模型版本，验证集准确率为 {best_accuracy * 100:.2f}%")
