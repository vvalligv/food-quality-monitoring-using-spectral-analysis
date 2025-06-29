# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import accuracy_score
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
# import matplotlib.pyplot as plt

# # Load the dataset from CSV file
# file_path = 'cleaned_dataset.csv'
# test_file_path="test_dataset.csv" # Replace with the path to your CSV file

# def load(train,test):
#   df = pd.read_csv(train)
#   test_df=pd.read_csv(test)
#   return df,test_df

# df,test_df=load(file_path,test_file_path)

# # Assuming 'att1' is the target variable
# # Specify the target column
# target_column = 'att1'
# test_target_c="att1"
# def shape(df,test_df,target_column,test_target_c):
#     target = df[target_column].values.reshape(-1, 1)
  
#     test_target= df[test_target_c].values.reshape(-1,1)
#     return target,test_target

# target,test_target=shape(df,test_df,target_column,test_target_c)

# # Use MinMaxScaler to scale the data between 0 and 1
# def scale(target,test_target):
#   scaler = MinMaxScaler(feature_range=(0, 1))
#   target_scaled = scaler.fit_transform(target)
#   scaler_test=MinMaxScaler(feature_range=(0,1))
#   test_scaled=scaler_test.fit_transform(test_target)
#   return target_scaled,test_scaled

# target_scaled,test_scaled=scale(target,test_target)
# # Define a function to create time series sequences
# def create_sequences(data, seq_length):
#     sequences = []
#     for i in range(len(data) - seq_length):
#         seq = data[i:i+seq_length]
#         sequences.append(seq)
#     return np.array(sequences)

# # Set a larger sequence length and split the data into training and testing sets
# sequence_length = 50  # Increase the sequence length
# X = create_sequences(target_scaled, sequence_length)
# y = target_scaled[sequence_length:]
# X_train = X
# y_train=y
# # Set a larger sequence length and split the data into training and testing sets
# sequence_length = 50  # Increase the sequence length
# X_t = create_sequences(test_scaled, sequence_length)
# y_t = test_scaled[sequence_length:]
# X_test = X_t
# y_test=y_t


# # Build a model with dilated convolutions to simulate TCN
# model = Sequential([
#     Conv1D(filters=128, kernel_size=3, activation='relu', dilation_rate=1, input_shape=(X_train.shape[1], X_train.shape[2])),
#     Conv1D(filters=128, kernel_size=3, activation='relu', dilation_rate=2),
#     Conv1D(filters=128, kernel_size=3, activation='relu', dilation_rate=4),
#     Conv1D(filters=128, kernel_size=3, activation='relu', dilation_rate=8),
#     GlobalAveragePooling1D(),
#     Dense(units=256, activation='relu'),
#     Dropout(0.5),
#     Dense(units=1, activation='sigmoid')
# ])

# # Learning rate schedule
# def lr_schedule(epoch):
#     lr = 0.0001 * np.exp(-0.05 * epoch)  # Adjust the learning rate schedule
#     return lr

# # Adjust learning rate
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# # Early stopping callback
# early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)  # Increase patience

# # Learning rate scheduler callback
# lr_scheduler = LearningRateScheduler(lr_schedule)

# # Lists to store accuracy and loss for each epoch
# epoch_accuracies = []
# epoch_losses = []

# # Train the model with more epochs
# for epoch in range(1, 151):  # Increase the number of epochs
#     history = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test), verbose=2, callbacks=[early_stopping, lr_scheduler])

#     # Make predictions
#     y_pred = model.predict(X_test)

#     # Define a threshold for classification
#     threshold = 0.5

#     # Convert predictions to binary classes
#     y_pred_binary = (y_pred > threshold).astype(int)
#     y_test_binary = (y_test > threshold).astype(int)

#     # Calculate accuracy
#     accuracy = accuracy_score(y_test_binary, y_pred_binary)
#     print(f'Accuracy on Test Set after Epoch {epoch}: {accuracy}')

#     # Append accuracy to the list
#     epoch_accuracies.append(accuracy)

#     # Append loss to the list
#     epoch_losses.append(history.history['val_loss'][0])

#     # Check if training should stop
#     if early_stopping.stopped_epoch > 0:
#         print(f"Training stopped at epoch {epoch} due to early stopping.")
#         break



# # Explicitly print final accuracy after the loop
# final_accuracy = epoch_accuracies[-10]
# print("Final Model Classification Accuracy: {:.2%}".format(final_accuracy))


# # Plot accuracy and loss for each epoch
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.plot(range(1, len(epoch_accuracies) + 1), epoch_accuracies, label='Accuracy')
# plt.title('Model Accuracy Over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label='Loss', color='red')
# plt.title('Model Loss Over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()

# plt.tight_layout()
# plt.show()

#ReLU activation function nnu sollum pothu, oru input value positive-a irundhaal, atha mathriya output kidaikkum. 
#Negative-a irundhaal, output-a 0 kidaikkum. For example, 5 input kudutha, 5-a than output kidaikkum. But -3 input kudutha, 0-a output kidaikkum.

#ReLU use panra reason, 
#athu simple-a irukkum, and it helps the network to learn complex patterns without vanishing gradients problem."**



import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

# Step 1: Load cleaned training dataset
df = pd.read_csv("final_cleaned_dataset.csv")  # Includes 'target' column

# Step 2: Split into features (X) and target (y)
X = df.drop(columns=['target'])
y = df['target']

# Step 3: Scale features using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)  # Fit on training data

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

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(64, 3, dilation_rate=1, activation='relu', padding='causal', input_shape=(sequence_length, X.shape[1])),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(64, 3, dilation_rate=2, activation='relu', padding='causal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(64, 3, dilation_rate=4, activation='relu', padding='causal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# Step 7: Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 8: Callbacks - EarlyStopping and LearningRateScheduler
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# def lr_schedule(epoch):
#     return 0.001 * (0.95 ** epoch)


lr_scheduler = LearningRateScheduler(lr_schedule)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)


# Step 9: Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stopping, lr_scheduler],
    verbose=2
)

# Step 10: Evaluate on validation set
y_val_pred_prob = model.predict(X_val)
y_val_pred = (y_val_pred_prob > 0.5).astype(int)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"\n✅ Validation Accuracy: {val_accuracy:.2%}")

# Step 11: Load and prepare external test dataset
test_df = pd.read_csv("cleaned_test_dataset.csv")
X_test_ext = test_df.drop(columns=['target'])
y_test_ext = test_df['target']

# Use the same scaler (fitted on training data)
X_test_ext_scaled = scaler.transform(X_test_ext)

# Create sequences from external test data
X_test_ext_seq, y_test_ext_seq = create_sequences(X_test_ext_scaled, y_test_ext.values, sequence_length)

# Step 12: Predict on external test data
y_test_ext_pred_prob = model.predict(X_test_ext_seq)
y_test_ext_pred = (y_test_ext_pred_prob > 0.5).astype(int)

# Accuracy on external test set
external_test_acc = accuracy_score(y_test_ext_seq, y_test_ext_pred)
print(f"\n📊 External Test Dataset Accuracy: {external_test_acc:.2%}")

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


