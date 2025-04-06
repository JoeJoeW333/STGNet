import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set font for Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Data path
data_path = 'data'

# Gesture classes
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

# Load data and unify time steps
def load_data(data_path, gestures, max_time_steps=100):
    X = []
    y = []
    for gesture_id, gesture in enumerate(gestures):
        gesture_folder = os.path.join(data_path, gesture)
        for file_name in os.listdir(gesture_folder):
            if file_name.endswith('.csv'):
                file_path = os.path.join(gesture_folder, file_name)
                data = pd.read_csv(file_path)

                # Use Range, Velocity, x, y features
                features = data[['Range', 'Velocity', 'x', 'y']].values
                X.append(features)
                y.append(gesture_id)
    
    # Unify time steps
    X = pad_sequences(X, maxlen=max_time_steps, dtype='float32', padding='post', truncating='post')
    return np.array(X), np.array(y)

# Load data
X, y = load_data(data_path, gestures, max_time_steps=100)

# One-hot encode labels
y = to_categorical(y, num_classes=len(gestures))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build CNN model
model = Sequential()
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(gestures), activation='softmax'))

# Compile model
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Set ModelCheckpoint callback
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# Set EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5000, restore_best_weights=True, mode='max', verbose=1)

# Train model
history = model.fit(X_train, y_train, epochs=1000, batch_size=128, validation_split=0.2, callbacks=[checkpoint, early_stopping])

# Load best model
model.load_weights('best_model.keras')

# Evaluate model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred_classes)
print(f"Overall accuracy: {accuracy * 100:.2f}%")

# Create output folder
output_folder = 'CNN-1000'
os.makedirs(output_folder, exist_ok=True)

# Plot confusion matrix (without percentage)
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=gestures, yticklabels=gestures)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(output_folder, 'confusion_matrix.pdf'), format='pdf', bbox_inches='tight')
plt.show()

# Plot confusion matrix (with percentage)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
cm_percent_str = np.array([[f"{value:.1f}%" for value in row] for row in cm_percent])

plt.figure(figsize=(10, 8))
sns.heatmap(cm_percent, annot=cm_percent_str, fmt='', cmap='Blues', xticklabels=gestures, yticklabels=gestures)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Percentage)')
plt.savefig(os.path.join(output_folder, 'confusion_matrix_percent.pdf'), format='pdf', bbox_inches='tight')
plt.show()

# Plot accuracy curve
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Curve')
plt.legend()
plt.savefig(os.path.join(output_folder, 'accuracy_curve.pdf'), format='pdf', bbox_inches='tight')
plt.show()

# Calculate and display per-class accuracy
train_pred = model.predict(X_train)
train_pred_classes = np.argmax(train_pred, axis=1)
train_true = np.argmax(y_train, axis=1)

train_cm = confusion_matrix(train_true, train_pred_classes)
train_class_accuracy = train_cm.diagonal() / train_cm.sum(axis=1)

test_class_accuracy = cm.diagonal() / cm.sum(axis=1)

# Create DataFrame
data = {
    'Gesture': gestures,
    'Training Accuracy': [f"{acc * 100:.2f}%" for acc in train_class_accuracy],
    'Testing Accuracy': [f"{acc * 100:.2f}%" for acc in test_class_accuracy]
}

df = pd.DataFrame(data)

# Print DataFrame
print(df.to_string(index=False))

# Plot per-class accuracy
fig, ax = plt.subplots(figsize=(12, 6))
width = 0.4

rects1 = ax.bar(df['Gesture'], [float(acc.strip('%')) for acc in df['Training Accuracy']], width, label='Training Accuracy')
rects2 = ax.bar([x + width for x in range(len(df['Gesture']))], [float(acc.strip('%')) for acc in df['Testing Accuracy']], width, label='Testing Accuracy')

ax.set_ylabel('Accuracy (%)')
ax.set_xlabel('Gesture')
ax.set_title('Training and Testing Accuracy for Each Gesture')
ax.set_xticklabels(df['Gesture'], rotation=45, ha="right")
ax.legend()

# Add horizontal dashed line
ax.axhline(y=90, color='r', linestyle='--', label='90% Accuracy Threshold')

# Add labels
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

# Calculate precision, recall, F1-score
precision = precision_score(y_true, y_pred_classes, average=None)
recall = recall_score(y_true, y_pred_classes, average=None)
f1 = f1_score(y_true, y_pred_classes, average=None)

# Create DataFrame
data = {
    'Gesture': gestures,
    'Precision (%)': [f"{p * 100:.2f}" for p in precision],
    'Recall (%)': [f"{r * 100:.2f}" for r in recall],
    'F1 Score (%)': [f"{f * 100:.2f}" for f in f1]
}

df = pd.DataFrame(data)

# Print DataFrame
print(df.to_string(index=False))
