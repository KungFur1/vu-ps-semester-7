import pandas
import os
import numpy as np


cancer_df = pandas.read_csv(os.path.join("breast-cancer-data", "breast-cancer-wisconsin.data"), header=None)

columns = [
    "Sample code number",
    "Clump Thickness",
    "Uniformity of Cell Size",
    "Uniformity of Cell Shape",
    "Marginal Adhesion",
    "Single Epithelial Cell Size",
    "Bare Nuclei",
    "Bland Chromatin",
    "Normal Nucleoli",
    "Mitoses",
    "Class"
]
cancer_df.columns = [col.lower().replace(' ', '_') for col in columns]
cancer_df[cancer_df.columns[10]] = cancer_df[cancer_df.columns[10]].apply(lambda x: 0 if x == 2 else 1) # 0 - begnin; 1 - malignant
cancer_df = cancer_df.replace('?', np.nan)
cancer_df = cancer_df.drop(cancer_df.columns[0], axis=1)
cancer_df = cancer_df.dropna()
cancer_df = cancer_df.sample(frac=1).reset_index(drop=True)

cancer_df.to_csv(os.path.join("refined-data", "breast-cancer.csv"), index=False)