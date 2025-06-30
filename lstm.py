
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

# Step 1: Load cleaned training dataset
df = pd.read_csv("final_cleaned_dataset.csv")  # Contains features + target

# Step 2: Split into features and target
X = df.drop(columns=['target'])
y = df['target']

# Step 3: Scale features using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)  # Fit on training set

# Step 4: Create sequences function
def create_sequences(X, y, seq_length=20):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])  # Predict next-step target
    return np.array(X_seq), np.array(y_seq)

sequence_length = 20
X_seq, y_seq = create_sequences(X_scaled, y.values, sequence_length)

# Step 5: Train-test split for training/validation
X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Step 6: Build the LSTM model
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(sequence_length, X.shape[1])),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.LSTM(100),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Step 7: Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 8: Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

def lr_schedule(epoch):
    return 0.001 * np.exp(-0.1 * epoch)

lr_scheduler = LearningRateScheduler(lr_schedule)

# Step 9: Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping, lr_scheduler],
    verbose=2
)

# Step 10: Evaluate on validation set
y_val_pred_prob = model.predict(X_val)
y_val_pred = (y_val_pred_prob > 0.5).astype(int)

val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"\n Validation Accuracy: {val_accuracy:.2%}")

# Step 11: Load and prepare external test dataset
test_df = pd.read_csv("cleaned_test_dataset.csv")
X_test_raw = test_df.drop(columns=['target'])
y_test_actual = test_df['target']

# Use the same scaler (fitted on training data)
X_test_scaled = scaler.transform(X_test_raw)

# Create sequences from external test data
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_actual.values, sequence_length)

# Step 12: Predict on external test data
y_test_pred_prob = model.predict(X_test_seq)
y_test_pred = (y_test_pred_prob > 0.5).astype(int)

# Accuracy on external test set
external_test_acc = accuracy_score(y_test_seq, y_test_pred)
print(f"\n External Test Dataset Accuracy: {external_test_acc:.2%}")

# Step 13: Plot Accuracy & Loss curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

