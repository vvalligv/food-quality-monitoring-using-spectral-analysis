#lstm model training
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler#This scaler from the sklearn.preprocessing module is used to scale features to a given range, usually [0, 1]. 
#It normalizes the data so that it fits within this range.

from sklearn.metrics import accuracy_score#This function from the sklearn.metrics module calculates the accuracy of a classification model.
from tensorflow. keras.callbacks import EarlyStopping, LearningRateScheduler
#Early stopping: is used to stop training when a monitored metric has stopped improving.
#It helps to prevent overfitting by stopping training when the model’s performance on the validation set starts to degrade.
#LearningRateScheduler: This callback allows you to adjust the learning rate of the model during training based on a given schedule or function.
#This can help improve model performance.
import matplotlib.pyplot as plt


#dataset reading
path = "cleaned_dataset.csv"
path1 = "test_dataset.csv"

# Load the dataset from CSV file
def load(path):
    df = pd.read_csv(path)
    return df  # Loads the csv file and returns as the dataframe

target_column = 'att1'
test_target_column = 'att1'

def shape(df,target_column):
    target = df[target_column].values.reshape(-1, 1)
    #Example: If target_column is 'age', df['age'] gives you the values of the 'age' column.
    # For .values:For instance, if df['age'] is [25, 30, 35], then df['age'].values would be array([25, 30, 35])
    #Example: If the array is [25, 30, 35], reshaping it with (-1, 1) will convert it to Reshaped 2D Column Vector:
#[[25]
 #[30]
 #[35]]
    #Using -1 means you don't have to manually calculate the number of rows, reducing the chance of errors and simplifying the code.
    return target

# Use MinMaxScaler to scale the data between 0 and 1

def scale(target,test_target):
  scaler = MinMaxScaler(feature_range=(0, 1))
  target_scaled = scaler.fit_transform(target)
  test_target_scaled = scaler.transform(test_target)
  return target_scaled, test_target_scaled

df = load(path)
test_df = load(path1)

target=shape(df,target_column)
test_target = shape(test_df,test_target_column)

target_scaled,test_target_scaled=scale(target,test_target)

# Define a function to create time series sequences
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    return np.array(sequences)

# Set the sequence length and split the data into training and testing sets
sequence_length = 20  # Adjust as needed
X = create_sequences(target_scaled, sequence_length)
y = target_scaled[sequence_length:]
X_train =X
y_train=y
X_test = create_sequences(test_target_scaled, sequence_length)
y_test = test_target_scaled[sequence_length:]

# Build a more complex LSTM model
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.Dropout(0.2),  # Adjust the dropout rate as needed
    tf.keras.layers.LSTM(units=100),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# Learning rate schedule
def lr_schedule(epoch):
    lr = 0.001 * np.exp(-0.1 * epoch)
    return lr

# Adjust learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Learning rate scheduler callback
lr_scheduler = LearningRateScheduler(lr_schedule)

# Lists to store accuracy and loss for each epoch
epoch_accuracies = []
epoch_losses = []

# Train the model with more epochs
for epoch in range(1, 101):  # Adjust the range based on the number of epochs
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
    print(f'Accuracy on : {accuracy}')

    # Append accuracy to the list
    epoch_accuracies.append(accuracy)

    # Append loss to the list
    epoch_losses.append(history.history['val_loss'][0])

    # Check if training should stop
    if early_stopping.stopped_epoch > 0:
        print(f"Training stopped at epoch {epoch} due to early stopping.")
        break

# Print final accuracy
final_accuracy = epoch_accuracies[-1]
print("Model Classification Accuracy: {:.2%}".format(final_accuracy))


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


