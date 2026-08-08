import pandas as pd
g = pd.read_parquet("data/goes_clean_1s.parquet")
print(g.dtypes)
print(g.head(3))
print("index:", g.index)