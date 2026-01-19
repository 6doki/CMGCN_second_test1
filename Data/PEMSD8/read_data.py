import numpy as np
import pandas as pd

data_file = "distance.csv"
with open(data_file, 'r') as f:
    data = pd.read_csv(f).values
print(max(data[:, 0]))
print(max(data[:, 1]))
