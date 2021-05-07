import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Cleaned-data-vc.csv")  # read data set

# print(df['vaxView'].value_counts())
fig = plt.figure(figsize=(10, 10))
scatter = plt.scatter(df['year'], df['value'], linewidths=1, alpha=.7, edgecolor='k', s=200, c=df['vaxView'])
plt.xlabel('Year')
plt.ylabel('Value')
plt.grid(True)
plt.show()

fig = plt.figure(figsize=(10, 10))
scatter = plt.scatter(df['sampleSize'], df['value'], linewidths=1, alpha=.7, edgecolor='k', s=200, c=df['vaxView'])
plt.xlim([20, 600])
plt.xlabel('sampleSize')
plt.ylabel('Value')
plt.grid(True)
plt.show()

fig = plt.figure(figsize=(10, 10))
scatter = plt.scatter(df['demographicClass'], df['value'], linewidths=1, alpha=.7, edgecolor='k', s=200, c=df['vaxView'])
plt.grid(True)
plt.xlabel('demographicClass')
plt.ylabel('Value')
plt.show()

# print(df['sampleSize'].value_counts())
# purple- label 0 for vaxView
# blue- label 1 for vaxView
# green- label 2 for vaxView
# yellow- label 3 for vaxView
