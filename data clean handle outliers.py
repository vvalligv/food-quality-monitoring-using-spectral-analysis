import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('output_dataset.csv')  # Replace 'your_dataset.csv' with the actual filename

# Print the shape of the original dataset
print("Original dataset shape:", data.shape)

# Drop duplicates
data.drop_duplicates(inplace=True)

# Handle missing values (if any)
data.dropna(inplace=True)

# Print the shape of the cleaned dataset
print("Cleaned dataset shape:", data.shape)

# Split the dataset into features and target variable
X = data.drop('target', axis=1)  # Features
y = data['target']  # Target variable

# Print the shape of features and target variable
print("X shape:", X.shape)
print("y shape:", y.shape)

# Handle outliers using the IQR method
def handle_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3.0 * IQR
    upper_bound = Q3 + 3.0 * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

# Create a copy of the original dataset for handling outliers
X_outliers_removed = X.copy()

# Handle outliers for each numerical column
numerical_columns = X.select_dtypes(include=['float64']).columns
for column in numerical_columns:
    X_outliers_removed = handle_outliers(X_outliers_removed, column)

# Print the shape of the dataset after handling outliers
print("Dataset shape after handling outliers:", X_outliers_removed.shape)

# Update target variable (y) based on the new shape of X
y = y.loc[X_outliers_removed.index]

# Print the cleaned dataset after handling outliers
print("\nCleaned Dataset after handling outliers:")
print(X_outliers_removed)

# Standardize numerical features (optional, depending on your model)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_outliers_removed)


# Now, X_train_scaled and X_test_scaled contain the standardized features,
# and y_train, y_test contain the target variable.

# Additional data cleaning or preprocessing steps can be added based on specific requirements.

# Display sample rows of X_train and y_train
print("\nSample X_train:")
print(X_train_scaled)

print("\nSample y_train:")
print(y.head())

# Save the cleaned dataset to a CSV file
X_outliers_removed.to_csv('cleaned_dataset.csv', index=False)
# Load the cleaned dataset from a CSV file
cleaned_data = pd.read_csv('cleaned_dataset.csv')

#y usually represents the target variable or dependent variable that your machine learning model aims to predict.
#X typically represents the feature matrix or set of independent variables in your machine learning dataset.
#The training dataset is used to train the machine learning model. It consists of a set of input-output pairs or examples.The test dataset is used to evaluate the
#performance of the trained machine learning model.
