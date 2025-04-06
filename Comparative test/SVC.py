import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils import class_weight

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

# 加载数据并统一特征长度
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

                # 如果时间步长不足，填充到固定长度
                if len(features) < max_time_steps:
                    padding = np.zeros((max_time_steps - len(features), features.shape[1]))
                    features = np.vstack((features, padding))
                else:
                    features = features[:max_time_steps]  # 截断到固定长度

                # 展平为特征向量
                features = features.flatten()
                X.append(features)
                y.append(gesture_id)
    
    return np.array(X), np.array(y)

# 加载数据
X, y = load_data(data_path, gestures, max_time_steps=100)

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 计算类别权重
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# 构建 SVC 模型
model = make_pipeline(StandardScaler(), SVC(kernel='rbf', class_weight=class_weights, probability=True))

# 训练模型
model.fit(X_train, y_train)

# 评估模型
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall accuracy: {accuracy * 100:.2f}%")

# 创建输出文件夹
output_folder = 'SVC'
os.makedirs(output_folder, exist_ok=True)

# 绘制混淆矩阵（不带百分比）
cm = confusion_matrix(y_test, y_pred)
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

# 计算并显示每个类别的准确率
train_pred = model.predict(X_train)
train_cm = confusion_matrix(y_train, train_pred)
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
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
f1 = f1_score(y_test, y_pred, average=None)

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
# ================== 新增部分结束 ==================