import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

df = pd.read_csv("Cleaned-data-vc.csv")

plt.hist(df['vaxView'])
plt.title("Range of vaxView", fontsize=14)
plt.xlabel("vaxView", fontsize=14)
plt.ylabel("No. of rows with that vaxView value", fontsize=14)
plt.show()

plt.scatter(df['vaxView'], df['value'])
plt.title("vaxView Vs Value", fontsize=14)
plt.xlabel("vaxView", fontsize=14)
plt.ylabel("value", fontsize=14)
plt.grid(True)
plt.show()

plt.hist(df['year'], bins=30)
plt.title('Range of year', fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.ylabel('No. of rows with that year', fontsize=14)
plt.show()

plt.scatter(df['year'], df['value'], marker='o')
plt.title('Year Vs Value', fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Value', fontsize=14)
plt.grid(True)
plt.show()

plt.hist(df['value'], bins=30)
plt.title('Range of value', fontsize=14)
plt.xlabel('Value', fontsize=14)
plt.ylabel('No. of rows with that value', fontsize=14)
plt.grid(True)
plt.show()

corr_matrix = df.corr()
sn.heatmap(corr_matrix, annot=True)
plt.show()
