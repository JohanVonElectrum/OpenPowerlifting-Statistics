import pandas as pd
import numpy as np

norm1 = np.random.normal(15, 4, 100)
norm2 = np.random.normal(3, 3, 100)
norm3 = norm1 + norm2

df = pd.DataFrame()

df["norm1"] = norm1
df["norm2"] = norm2
df["norm3"] = norm3

df.to_csv("norm.csv")