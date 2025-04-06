import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Conv1D, MaxPooling1D, Flatten, Add, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据路径
data_path = 'data'

# 手势类别
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

# 加载数据并统一时间步长
def load_data(data_path, gestures, max_time_steps=100):
    X = []
    y = []
    for gesture_id, gesture in enumerate(gestures):
        gesture_folder = os.path.join(data_path, gesture)
        for file_name in os.listdir(gesture_folder):
            if file_name.endswith('.csv'):
                file_path = os.path.join(gesture_folder, file_name)
                data = pd.read_csv(file_path)

                # 使用 Range, Velocity, x, y 特征
                features = data[['Range', 'Velocity', 'x', 'y']].values
                X.append(features)
                y.append(gesture_id)
    
    # 统一时间步长
    X = pad_sequences(X, maxlen=max_time_steps, dtype='float32', padding='post', truncating='post')
    return np.array(X), np.array(y)

# 加载数据
X, y = load_data(data_path, gestures, max_time_steps=100)

# 对标签进行 one-hot 编码
y = to_categorical(y, num_classes=len(gestures))

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义残差块
def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x

    x = Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

# 构建 ResNet 模型
input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
x = Conv1D(64, 7, padding='same')(input_layer)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling1D(3)(x)

x = residual_block(x, 64)
x = residual_block(x, 64)
x = residual_block(x, 128, stride=2)
x = residual_block(x, 128)
x = residual_block(x, 256, stride=2)
x = residual_block(x, 256)

x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(len(gestures), activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)

# 编译模型
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# 设置 ModelCheckpoint 回调
checkpoint = ModelCheckpoint('best_model_resnet.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# 设置 EarlyStopping 回调
early_stopping = EarlyStopping(monitor='val_accuracy', patience=8000, restore_best_weights=True, mode='max', verbose=1)

# 训练模型
history = model.fit(X_train, y_train, epochs=10000, batch_size=128, validation_split=0.2, callbacks=[checkpoint, early_stopping])

# 加载最佳模型
model.load_weights('best_model_resnet.keras')

# 评估模型
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# 计算准确率
accuracy = accuracy_score(y_true, y_pred_classes)
print(f"Overall accuracy: {accuracy * 100:.2f}%")

# 创建输出文件夹
output_folder = 'ResNet-10000'
os.makedirs(output_folder, exist_ok=True)

# 绘制混淆矩阵（不带百分比）
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=gestures, yticklabels=gestures)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(output_folder, 'confusion_matrix.pdf'), format='pdf', bbox_inches='tight')
plt.show()

# 绘制混淆矩阵（带百分比）
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
cm_percent_str = np.array([[f"{value:.1f}%" for value in row] for row in cm_percent])

plt.figure(figsize=(10, 8))
sns.heatmap(cm_percent, annot=cm_percent_str, fmt='', cmap='Blues', xticklabels=gestures, yticklabels=gestures)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Percentage)')
plt.savefig(os.path.join(output_folder, 'confusion_matrix_percent.pdf'), format='pdf', bbox_inches='tight')
plt.show()

# 绘制准确率曲线
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Curve')
plt.legend()
plt.savefig(os.path.join(output_folder, 'accuracy_curve.pdf'), format='pdf', bbox_inches='tight')
plt.show()

# 计算并显示每个类别的准确率
train_pred = model.predict(X_train)
train_pred_classes = np.argmax(train_pred, axis=1)
train_true = np.argmax(y_train, axis=1)

train_cm = confusion_matrix(train_true, train_pred_classes)
train_class_accuracy = train_cm.diagonal() / train_cm.sum(axis=1)

test_class_accuracy = cm.diagonal() / cm.sum(axis=1)

# 创建 DataFrame
data = {
    'Gesture': gestures,
    'Training Accuracy': [f"{acc * 100:.2f}%" for acc in train_class_accuracy],
    'Testing Accuracy': [f"{acc * 100:.2f}%" for acc in test_class_accuracy]
}

df = pd.DataFrame(data)

# 打印 DataFrame
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
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8)

autolabel(rects1)
autolabel(rects2)

ax.legend()
fig.tight_layout()
plt.savefig(os.path.join(output_folder, 'class_accuracy.pdf'), format='pdf', bbox_inches='tight')
plt.show()

# 计算 Precision、Recall、F1-score
precision = precision_score(y_true, y_pred_classes, average=None)
recall = recall_score(y_true, y_pred_classes, average=None)
f1 = f1_score(y_true, y_pred_classes, average=None)

# 创建 DataFrame
data = {
    'Gesture': gestures,
    'Precision (%)': [f"{p * 100:.2f}" for p in precision],
    'Recall (%)': [f"{r * 100:.2f}" for r in recall],
    'F1 Score (%)': [f"{f * 100:.2f}" for f in f1]
}

df = pd.DataFrame(data)

# 打印 DataFrame
print(df.to_string(index=False))

# ================== 新增部分：生成两个 CSV 文件 ==================
# 保存每个类别的准确率到 CSV 文件
accuracy_csv_path = os.path.join(output_folder, 'class_accuracy.csv')
pd.DataFrame({
    'Gesture': gestures,
    'Training Accuracy': [f"{acc * 100:.2f}%" for acc in train_class_accuracy],
    'Testing Accuracy': [f"{acc * 100:.2f}%" for acc in test_class_accuracy]
}).to_csv(accuracy_csv_path, index=False)

# 保存 Precision、Recall、F1-score 到 CSV 文件
metrics_csv_path = os.path.join(output_folder, 'classification_metrics.csv')
pd.DataFrame({
    'Gesture': gestures,
    'Precision (%)': [f"{p * 100:.2f}" for p in precision],
    'Recall (%)': [f"{r * 100:.2f}" for r in recall],
    'F1 Score (%)': [f"{f * 100:.2f}" for f in f1]
}).to_csv(metrics_csv_path, index=False)

print(f"Class accuracy saved to {accuracy_csv_path}")
print(f"Classification metrics saved to {metrics_csv_path}")