import pandas as pd
import matplotlib.pyplot as plt

# Replace 'your_file.csv' with the actual path to your CSV file
df = pd.read_csv('output_dataset.csv')

# Plot histograms for the first 5 columns
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(4, 8))

for i in range(5):
    axes[i].hist(df.iloc[:, i], bins=20, color='blue', edgecolor='black')
    axes[i].set_title(df.columns[i])
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()


