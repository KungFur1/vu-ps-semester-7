import pandas
import os
import numpy as np


# Iris - encode last column and name columns
iris_df = pandas.read_csv(os.path.join("iris-data", "iris.data"), header=None, index_col=False)

iris_df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_df['species'] = iris_df['species'].apply(lambda x: 0 if x == "Iris-versicolor" 
                                                else 1 if x == "Iris-virginica" 
                                                else 2 if x == "Iris-setosa"
                                                else 3)

iris_df = iris_df[iris_df['species'].isin([0, 1])]
iris_df = iris_df.sample(frac=1).reset_index(drop=True)

iris_df.to_csv(os.path.join("refined-data", "iris.csv"), index=False)


# Iris - add artificial data-points (by generating random features between min & max values)
iris_df = pandas.read_csv(os.path.join("refined-data", "iris.csv"))

grouped_df = iris_df.groupby('species')
min_df = grouped_df.min()
max_df = grouped_df.max()

artificial_samples = []
for (group_key, min_row), (_, max_row) in zip(min_df.iterrows(), max_df.iterrows()):
    for i in range(200):
        new_sample = {'species': group_key}

        for column in min_df.columns:
            min_val = min_row[column]
            max_val = max_row[column]
            random_val = np.round(np.random.uniform(low=min_val, high=max_val), 1)
            new_sample[column] = random_val
        
        artificial_samples.append(new_sample)

artificial_df = pandas.DataFrame(artificial_samples)
artificial_df = artificial_df.sample(frac=1).reset_index(drop=True)
artificial_df.to_csv(os.path.join("refined-data", "iris-artificial.csv"), index=False)


# Iris - Combine artificial and real data
iris_df = pandas.read_csv(os.path.join("refined-data", "iris.csv"))
artificial_df = pandas.read_csv(os.path.join("refined-data", "iris-artificial.csv"))

combined_df = pandas.concat([iris_df, artificial_df], ignore_index=True)
combined_df = combined_df.sample(frac=1).reset_index(drop=True)

combined_df.to_csv(os.path.join("refined-data", "iris-combined.csv"), index=False)
