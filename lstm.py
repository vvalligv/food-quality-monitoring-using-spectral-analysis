# #lstm model training
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler#This scaler from the sklearn.preprocessing module is used to scale features to a given range, usually [0, 1]. 
# #It normalizes the data so that it fits within this range.

# from sklearn.metrics import accuracy_score#This function from the sklearn.metrics module calculates the accuracy of a classification model.
# from tensorflow. keras.callbacks import EarlyStopping, LearningRateScheduler
# #Early stopping: is used to stop training when a monitored metric has stopped improving.
# #It helps to prevent overfitting by stopping training when the model’s performance on the validation set starts to degrade.
# #LearningRateScheduler: This callback allows you to adjust the learning rate of the model during training based on a given schedule or function.
# #This can help improve model performance.
# import matplotlib.pyplot as plt


# #dataset reading
# path = "cleaned_dataset.csv"
# path1 = "test_dataset.csv"  # first, in training dtataset it is trained, and then in test dataset, it is evaluated.

# # Load the dataset from CSV file
# def load(path):
#     df = pd.read_csv(path)
#     return df  # Loads the csv file and returns as the dataframe

# target_column = 'att1'
# test_target_column = 'att1'

# def shape(df,target_column):
#     target = df[target_column].values.reshape(-1, 1)
#     #Example: If target_column is 'age', df['age'] gives you the values of the 'age' column.
#     # For .values:For instance, if df['age'] is [25, 30, 35], then df['age'].values would be array([25, 30, 35])
#     #Example: If the array is [25, 30, 35], reshaping it with (-1, 1) will convert it to Reshaped 2D Column Vector:
# #[[25]
#  #[30]
#  #[35]]

#     #| Code             | Output Shape | Meaning                            |
# | ---------------- | ------------ | ---------------------------------- |
# #| `reshape(-1, 1)` | (n, 1)       | Column vector                      |
# #| `reshape(1, -1)` | (1, n)       | Row vector                         |
# #| `reshape(n,)`    | (n,)         | 1D vector (original shape)         |
# #| `reshape(3, 2)`  | (3, 2)       | Reshape manually to 3 rows, 2 cols |

#     #Using -1 means you don't have to manually calculate the number of rows, reducing the chance of errors and simplifying the code.
#     return target   # Dataset load pannina apram, target column-a reshape pannina, model ku correct-a format-la data kudukalam. 
#     #Ippo, scaling or training ellam easy-a nadakum."

# # Use MinMaxScaler to scale the data between 0 and 1

# def scale(target,test_target):
#   scaler = MinMaxScaler(feature_range=(0, 1))
#   target_scaled = scaler.fit_transform(target)
#   test_target_scaled = scaler.transform(test_target)
#   return target_scaled, test_target_scaled    #Training data-a scale pannitu, adhe scaling-a eduthutu test data-kum apply pannum.
#    # Ippo rendu datasets-um 0 to 1 range-la irukum, which makes it easier for the LSTM model to train and predict correctly.

# df = load(path)
# test_df = load(path1)

# target=shape(df,target_column)
# test_target = shape(test_df,test_target_column)

# target_scaled,test_target_scaled=scale(target,test_target)

# # Define a function to create time series sequences
# def create_sequences(data, seq_length):
#     sequences = []
#     for i in range(len(data) - seq_length):
#         seq = data[i:i+seq_length]
#         sequences.append(seq)
#     return np.array(sequences)

# # Set the sequence length and split the data into training and testing sets
# sequence_length = 20  # Adjust as needed
# X = create_sequences(target_scaled, sequence_length)
# y = target_scaled[sequence_length:]
# X_train =X
# y_train=y
# X_test = create_sequences(test_target_scaled, sequence_length)
# y_test = test_target_scaled[sequence_length:]

# # Build a more complex LSTM model
# model = tf.keras.models.Sequential([
#     tf.keras.layers.LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
#     tf.keras.layers.Dropout(0.2),  # Adjust the dropout rate as needed
#     tf.keras.layers.LSTM(units=100),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(units=1, activation='sigmoid')
# ])

# # Learning rate schedule
# def lr_schedule(epoch):
#     lr = 0.001 * np.exp(-0.1 * epoch)
#     return lr

# # Adjust learning rate
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# # Early stopping callback
# early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# # Learning rate scheduler callback
# lr_scheduler = LearningRateScheduler(lr_schedule)

# # Lists to store accuracy and loss for each epoch
# epoch_accuracies = []
# epoch_losses = []

# # Train the model with more epochs
# for epoch in range(1, 101):  # Adjust the range based on the number of epochs
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
#     print(f'Accuracy on : {accuracy}')

#     # Append accuracy to the list
#     epoch_accuracies.append(accuracy)

#     # Append loss to the list
#     epoch_losses.append(history.history['val_loss'][0])

#     # Check if training should stop
#     if early_stopping.stopped_epoch > 0:
#         print(f"Training stopped at epoch {epoch} due to early stopping.")
#         break

# # Print final accuracy
# final_accuracy = epoch_accuracies[-1]
# print("Model Classification Accuracy: {:.2%}".format(final_accuracy))


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

