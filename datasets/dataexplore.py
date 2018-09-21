import pandas as pd
import numpy as np


print("loading the CSV")
# set indexes?
try:
    df = pd.read_csv("articles_combined.csv")
except Exception:
    print(
        "Some problem loading the articles_combined.csv - you need to unzip your csv and run combine.py first!"
    )


print("describe ----")
print(df.describe())
print("columns ----")
print(df.columns)
print("top and bottom 10 rows ----")
print(df[["date", "title", "content", "publication"]].head(10))
print(df[["date", "title", "content", "publication"]].tail(10))
