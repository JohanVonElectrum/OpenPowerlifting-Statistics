import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats
from typing import Callable, Any
from tqdm import tqdm

print("Importing dataset...")
df = pd.read_csv("dataset.csv", low_memory=False)
print("Dataset rows:", df.shape[0])
df = df.loc[:, df.isnull().mean() < .8].dropna()
df["WeightClassKg"] = pd.to_numeric(df["WeightClassKg"], downcast="float", errors="coerce")
dataframe: pd.DataFrame = df[["Sex", "Age", "WeightClassKg", "BodyweightKg", "TotalKg", "Wilks"]].dropna()
print("Dataset rows after clean:", df.shape[0])

weights = np.histogram(dataframe["WeightClassKg"], bins=10)[1]
for index, row in tqdm(dataframe.iterrows()):
    for i in range(1, len(weights) - 1):
        if row["WeightClassKg"] < weights[i]:
            dataframe.at[index, "WeightClassKg"] = f"[{weights[i - 1]}, {weights[i]})"
            break
        dataframe.at[index, "WeightClassKg"] = f"[{weights[len(weights) - 2]}, {weights[len(weights) - 1]}]"


dataframe.to_csv("clean.csv")
reduced: pd.DataFrame = dataframe.drop(np.random.choice(dataframe.index, len(dataframe.index) - 1000, replace=False))
reduced.to_csv("reduced.csv")

for name in tqdm(dataframe.keys()):
    unique = np.sort(dataframe[name].unique())
    if len(unique) <= 10:
        print(name, f"(D): {unique}")
    else:
        mean = dataframe[name].mean()
        std = dataframe[name].std()
        skew = dataframe[name].skew()
        kurt = dataframe[name].kurt()
        print(name, f"(C{' NORM' if stats.kstest((dataframe[name] - mean) / std, 'norm')[1] > 0.05 else ''}):\n\tmean={mean}\n\trange=[{dataframe[name].min()}, {dataframe[name].max()}]\n\tstd={std}\n\tskew={skew}\n\tkurt={kurt}")
        plt.hist(dataframe[name])
        plt.title(name)
        plt.show()


for name in tqdm(reduced.keys()):
    unique = np.sort(reduced[name].unique())
    if len(unique) <= 10:
        print(name, f"(D): {unique}")
    else:
        mean = reduced[name].mean()
        std = reduced[name].std()
        skew = reduced[name].skew()
        kurt = reduced[name].kurt()
        print(name, f"(C{' NORM' if stats.kstest((reduced[name] - mean) / std, 'norm')[1] > 0.05 else ''}):\n\tmean={mean}\n\trange=[{reduced[name].min()}, {reduced[name].max()}]\n\tstd={std}\n\tskew={skew}\n\tkurt={kurt}")
        plt.hist(reduced[name])
        plt.title(name)
        plt.show()

