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
    exit()


print("describe ----")
print(df.describe())

print("columns ----")
print(df.columns)

print("top and bottom 10 rows ----")
# print(df[["date", "title", "content", "publication"]].head(10))
print(df[:10][["title"]])
print(df[["date", "title", "content", "publication"]].tail(10))

print("headline and content from row 1 ----")
print("= = = = = = = = = = = = = = = = = =")
print(df.loc[df.index[0], "title"])
print("- - - - - - - - - - - - - - - - - -")
print(df.loc[df.index[0], "content"])
print("= = = = = = = = = = = = = = = = = =")


print("generate msgpack from dataframe")
df.to_msgpack("./tester.pack")

print("loading the dataframe back from the file")
df_new = pd.read_msgpack("./tester.pack")
print(df_new.describe())

