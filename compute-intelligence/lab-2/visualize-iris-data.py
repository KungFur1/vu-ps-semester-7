import pandas as pd
import matplotlib.pyplot as plt
import os

# df = pd.read_csv(os.path.join("refined-data", "iris.csv"))
df = pd.read_csv(os.path.join("refined-data", "iris.csv"))
colors = {0: 'red', 1: 'blue', 2: 'green'}


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for species, color in colors.items():
    subset = df[df['species'] == species]
    axes[0].scatter(subset['sepal_length'], subset['sepal_width'], label=f"Species {species}", color=color)
axes[0].set_xlabel('Sepal Length')
axes[0].set_ylabel('Sepal Width')
axes[0].set_title('Sepal Length vs Sepal Width')
axes[0].legend()
for species, color in colors.items():
    subset = df[df['species'] == species]
    axes[1].scatter(subset['petal_length'], subset['petal_width'], label=f"Species {species}", color=color)
axes[1].set_xlabel('Petal Length')
axes[1].set_ylabel('Petal Width')
axes[1].set_title('Petal Length vs Petal Width')
axes[1].legend()
plt.tight_layout()
plt.show()