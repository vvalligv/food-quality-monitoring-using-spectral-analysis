import pandas as pd
from sklearn.model_selection import train_test_split # sklearn is a library from where train_test_split func is imported for splitting train and test dataset
from sklearn.preprocessing import StandardScaler #StandardScaler used for standardising values

#Scikit-learn is essential for building machine learning models and performing data analysis.
#It provides tools for preprocessing, model selection, classification, regression, clustering, dimensionality reduction, and model evaluation.
#The library is designed to be easy to use and integrate well with other Python scientific computing libraries.

# Load the dataset
data = pd.read_csv('output_dataset.csv')  # Replace 'your_dataset.csv' with the actual filename

# Print the shape of the original dataset
print("Original dataset shape:", data.shape)#original dataset shape is printed like how many rows,how many columns

# Drop duplicates
data.drop_duplicates(inplace=True)#used for dropping duplicates rows.The inplace=True argument ensures that the changes are made to the DataFrame itself and nothing is returned.
#if inplace=False, the method will not modify the original DataFrame. Instead, it will return a new DataFrame with duplicates removed, leaving the original DataFrame unchanged.

# Handle missing values (if any)
data.dropna(inplace=True) # if any missing values are found, it means,that row(missing value rows) will be deleted from dataset.

# Print the shape of the cleaned dataset
print("Cleaned dataset shape:", data.shape)#Cleared dataset shape will be displayed

# Split the dataset into features and target variable
X = data.drop('target', axis=1)  # Features # Here, dataset is transferred completely except target column to X variable, where X is a dataframe.
#axis=1: Indicates that we're dropping a column (not a row). If axis=0, it would drop rows instead.

y = data['target']  # Target variable #Target column is stored in y variable

# Print the shape of features and target variable
print("X shape:", X.shape)#X shape, indicates rows and columns --> feature shape
print("y shape:", y.shape)#Y shape, indicates Y shape --> target variable shape

# Handle outliers using the IQR method
def handle_outliers(df, column):
    Q1 = df[column].quantile(0.25)  
    Q3 = df[column].quantile(0.75)
    
    #Q1 Position: 0.25 * (N + 1) where N is the number of data points.
    #Q3 Position: 0.75 * (N + 1)
    #Interpolate Values  Example:

    #Q1: The position 2.75 is between the 2nd and 3rd values. Interpolate to find Q1:

   # 2nd value = 12
   # 3rd value = 14
  #  Q1 = 12 + 0.75 * (14 - 12) = 12 + 1.5 = 13.5
  #  Q3: The position 8.25 is between the 8th and 9th values. Interpolate to find Q3:

  #  8th value = 22
  #  9th value = 24
 #   Q3 = 22 + 0.25 * (24 - 22) = 22 + 0.5 = 22.5
    #Example over

    IQR = Q3 - Q1
    
#Why Use 3.0 Instead of 1.5:

#More Stringent Criteria: Using a multiplier of 3.0 makes the criteria for identifying outliers more stringent. It expands the range of what is considered "normal" and thus identifies fewer extreme values as outliers.
#Context-Specific: The factor of 3.0 is sometimes used in specific contexts where:
#Data Distribution: The data is expected to have fewer extreme outliers.
#Robust Analysis: The analysis requires a stricter definition of outliers to avoid including potentially influential points that could skew results.

    lower_bound = Q1 - 3.0 * IQR
    upper_bound = Q3 + 3.0 * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)] # values below lower bound and values above upper bound are outliers
    return df

# Create a copy of the original dataset for handling outliers
X_outliers_removed = X.copy() 

# Handle outliers for each numerical column
numerical_columns = X.select_dtypes(include=['float64']).columns #Numerical columns are identified
for column in numerical_columns: # each numerical column is iterated
    X_outliers_removed = handle_outliers(X_outliers_removed, column)--> #Purpose: This line calls the handle_outliers function for the current column being processed, and updates X_outliers_removed with the result.
#handle_outliers(X_outliers_removed, column): This function is designed to identify and remove outliers in the specified column of the DataFrame X_outliers_removed.

# Print the shape of the dataset after handling outliers
print("Dataset shape after handling outliers:", X_outliers_removed.shape) # After handling outliers , that shape will be printed

# Update target variable (y) based on the new shape of X
y = y.loc[X_outliers_removed.index]
#example
#Updated Target Variable y:
#Original y: [0, 1, 0, 1, 1]
#Indices of X_outliers_removed: Suppose indices [0, 1, 2, 3] remain.
#Updated y: After y.loc[X_outliers_removed.index], y will be [0, 1, 0, 1] (values corresponding to indices [0, 1, 2, 3]).

# Print the cleaned dataset after handling outliers
print("\nCleaned Dataset after handling outliers:")
print(X_outliers_removed)



# Standardize numerical features (optional, depending on your model)
scaler = StandardScaler()
#Create an instance of the StandardScaler class from sklearn.preprocessing.
#This object will be used to compute the mean and standard deviation of the features and then use these statistics to transform the data.

#example 
#First, mean and standard deviation is calculated for each column, 
#and then: each value - mean/standard deviation. is done

X_train_scaled = scaler.fit_transform(X_outliers_removed) #This method simultaneously fits the scaler to the data (calculates the necessary statistics) and applies the transformation to the data.
#fit(): Computes the mean and standard deviation for each numerical feature in X_outliers_removed.
#transform(): Uses the computed mean and standard deviation to scale each feature such that the resulting feature values have a mean of 0 and a standard deviation of 1.
#X_train_scaled will contain the standardized version of X_outliers_removed, where each feature is scaled accordingly.


# Now, X_train_scaled and X_test_scaled contain the standardized features,
# and y_train, y_test contain the target variable.

# Additional data cleaning or preprocessing steps can be added based on specific requirements.

# Display sample rows of X_train and y_train
print("\nSample X_train:")
print(X_train_scaled)
#Since X_train_scaled is a NumPy array (as returned by fit_transform), the printed output will be in array format. 
#Each row corresponds to an observation, and each column corresponds to a feature.

print("\nSample y_train:")
print(y.head())

# Save the cleaned dataset to a CSV file
X_outliers_removed.to_csv('cleaned_dataset.csv', index=False)

#index=False: Excludes row indices from the CSV file.
#index=True: Includes row indices in the CSV file.

# Load the cleaned dataset from a CSV file
cleaned_data = pd.read_csv('cleaned_dataset.csv')

#y usually represents the target variable or dependent variable that your machine learning model aims to predict.
#X typically represents the feature matrix or set of independent variables in your machine learning dataset.
#The training dataset is used to train the machine learning model. It consists of a set of input-output pairs or examples.The test dataset is used to evaluate the
#performance of the trained machine learning model.
