#Both pandas and matplotlib  are libraries
import pandas as pd  # Used for data management
import matplotlib.pyplot as plt #Used for drawing graphs,charts etc.

# Replace 'your_file.csv' with the actual path to your CSV file
df = pd.read_csv('output_dataset.csv') # Here, read_csv is a function in pandas to load the data which is in output_dataset and we store it in a dataframe called as df

# Plot histograms for the first 5 columns
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(4, 8)) # Here, we use plt.subplots func, to plot in size of 4 inches wide ,8 inches long fig size and here, 5 rows and i column is plotted.
#fig --> It is a variable which holds all the plot,inclu label,title etc.
#axes --> This variable is an array of subplot objects (or a single subplot if only one subplot is created).eaxh subplot is a axes

for i in range(5): # Here, we use for loop for plotting 5 columns from dataset using histogram
    axes[i].hist(df.iloc[:, i], bins=20, color='blue', edgecolor='black') # df.iloc[:, i]--> : This selects the i-th column of the DataFrame df(0 to 4)
    #Here ,we use those 5 columns from dataset to draw hist,bins=20 different ranges of values to plot histogram,i.e 20 bins or 20 bars
    axes[i].set_title(df.columns[i]) --># Use this to set title for the plots for particular columns eg, attr1,attr2,etc
    axes[i].set_xlabel('Value') --> #x-axis --> value
    axes[i].set_ylabel('Frequency')--> #y-axies --> Frequency

plt.tight_layout()--> #make graphs not to overlap
plt.show()-->#shows the plot

#Histograms are essential for understanding the distribution
#and frequency of numerical data, helping to visualize and analyze
#patterns, compare different datasets, and identify any anomalies or outliers.
