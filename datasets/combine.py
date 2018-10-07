import pandas as pd
import numpy as np

import glob, os, time


def read_and_describe_csv(filename):
    # dates are in the form 2017-01-05
    to_datetime = lambda d: time.strptime(d.replace("/", "-"), "%Y-%m-%d")
    replace_smartquotes = (
        lambda x: x.replace("“", '"')
        .replace("”", '"')
        .replace("‟", '"')
        .replace("‶", '"')
        .replace("″", '"')
        .replace("’", "'")
        .replace("‵", "'")
        .replace("‘", "'")
        .replace("’", "'")
        .replace("′", "'")
        .replace("‛", "'")
    )
    print("- - - - - - - - - - - - - - - -")
    print('loading file "%s"' % filename)
    df = pd.read_csv(
        filename,
        dtype={
            "id": np.int64,
            "title": np.str,
            "publication": np.str,
            "author": np.str,
            "content": np.str,
            # "date": np.str,
            "year": np.float,
            "month": np.float,
        },
        header=0,
        names=[
            "ignore",
            "id",
            "title",
            "publication",
            "author",
            "date",
            "year",
            "month",
            "url",
            "content",
        ],
    )

    print("Dropping columns we don't want - unnamed, url")
    df = df.drop(["ignore", "url"], axis=1)
    print("Columns now : ", df.columns)

    print("Removing bad rows - titles and dates missing on some rows")
    df = df[df["title"].apply(lambda x: type(x) == str)]
    df = df[df["date"].apply(lambda x: type(x) == str)]

    print("Replacing smartquotes with straight quotes")
    df["title"] = df.title.apply(replace_smartquotes)
    df["content"] = df.content.apply(replace_smartquotes)

    # print("Converting date to datetime")
    # df.date = df.date.apply(to_datetime)
    # print("a few examples of the date now", df.date.head(10))

    return df


df = pd.concat([read_and_describe_csv(f) for f in glob.glob("articles*.csv")])

print("Combined articles")
# print(df.describe())
print(df.columns)

df.to_csv("combined_articles.csv")
df.to_msgpack("./combined_articles.pack")
