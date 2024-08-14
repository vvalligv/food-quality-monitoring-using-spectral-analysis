import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import matplotlib.pyplot as plt

# Load the dataset from CSV file
file_path = 'cleaned_dataset.csv'
test_file_path="test_dataset.csv" # Replace with the path to your CSV file

def load(train,test):
  df = pd.read_csv(train)
  test_df=pd.read_csv(test)
  return df,test_df

df,test_df=load(file_path,test_file_path)

# Assuming 'att1' is the target variable
# Specify the target column
target_column = 'att1'
test_target_c="att1"
def shape(df,test_df,target_column,test_target_c):
    target = df[target_column].values.reshape(-1, 1)
  
    test_target= df[test_target_c].values.reshape(-1,1)
    return target,test_target

target,test_target=shape(df,test_df,target_column,test_target_c)

# Use MinMaxScaler to scale the data between 0 and 1
def scale(target,test_target):
  scaler = MinMaxScaler(feature_range=(0, 1))
  target_scaled = scaler.fit_transform(target)
  scaler_test=MinMaxScaler(feature_range=(0,1))
  test_scaled=scaler_test.fit_transform(test_target)
  return target_scaled,test_scaled

target_scaled,test_scaled=scale(target,test_target)
# Define a function to create time series sequences
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    return np.array(sequences)

# Set a larger sequence length and split the data into training and testing sets
sequence_length = 50  # Increase the sequence length
X = create_sequences(target_scaled, sequence_length)
y = target_scaled[sequence_length:]
X_train = X
y_train=y
# Set a larger sequence length and split the data into training and testing sets
sequence_length = 50  # Increase the sequence length
X_t = create_sequences(test_scaled, sequence_length)
y_t = test_scaled[sequence_length:]
X_test = X_t
y_test=y_t


# Build a model with dilated convolutions to simulate TCN
model = Sequential([
    Conv1D(filters=128, kernel_size=3, activation='relu', dilation_rate=1, input_shape=(X_train.shape[1], X_train.shape[2])),
    Conv1D(filters=128, kernel_size=3, activation='relu', dilation_rate=2),
    Conv1D(filters=128, kernel_size=3, activation='relu', dilation_rate=4),
    Conv1D(filters=128, kernel_size=3, activation='relu', dilation_rate=8),
    GlobalAveragePooling1D(),
    Dense(units=256, activation='relu'),
    Dropout(0.5),
    Dense(units=1, activation='sigmoid')
])

# Learning rate schedule
def lr_schedule(epoch):
    lr = 0.0001 * np.exp(-0.05 * epoch)  # Adjust the learning rate schedule
    return lr

# Adjust learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)  # Increase patience

# Learning rate scheduler callback
lr_scheduler = LearningRateScheduler(lr_schedule)

# Lists to store accuracy and loss for each epoch
epoch_accuracies = []
epoch_losses = []

# Train the model with more epochs
for epoch in range(1, 151):  # Increase the number of epochs
    history = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test), verbose=2, callbacks=[early_stopping, lr_scheduler])

    # Make predictions
    y_pred = model.predict(X_test)

    # Define a threshold for classification
    threshold = 0.5

    # Convert predictions to binary classes
    y_pred_binary = (y_pred > threshold).astype(int)
    y_test_binary = (y_test > threshold).astype(int)

    # Calculate accuracy
    accuracy = accuracy_score(y_test_binary, y_pred_binary)
    print(f'Accuracy on Test Set after Epoch {epoch}: {accuracy}')

    # Append accuracy to the list
    epoch_accuracies.append(accuracy)

    # Append loss to the list
    epoch_losses.append(history.history['val_loss'][0])

    # Check if training should stop
    if early_stopping.stopped_epoch > 0:
        print(f"Training stopped at epoch {epoch} due to early stopping.")
        break



# Explicitly print final accuracy after the loop
final_accuracy = epoch_accuracies[-10]
print("Final Model Classification Accuracy: {:.2%}".format(final_accuracy))


# Plot accuracy and loss for each epoch
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(epoch_accuracies) + 1), epoch_accuracies, label='Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label='Loss', color='red')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

#ReLU activation function nnu sollum pothu, oru input value positive-a irundhaal, atha mathriya output kidaikkum. 
#Negative-a irundhaal, output-a 0 kidaikkum. For example, 5 input kudutha, 5-a than output kidaikkum. But -3 input kudutha, 0-a output kidaikkum.

#ReLU use panra reason, 
#athu simple-a irukkum, and it helps the network to learn complex patterns without vanishing gradients problem."**
