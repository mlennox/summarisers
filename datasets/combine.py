import pandas as pd
import numpy as np

import glob, os


def read_and_describe_csv(filename):
    print('loading file "%s"' % filename)
    df = pd.read_csv(
        filename,
        # "./%s.csv" % filename,
        # index_col="id",
        date_parser=pd.to_datetime,
        dtype={
            "id": np.int64,
            "title": np.str,
            "publication": np.str,
            "author": np.str,
            "content": np.str,
            # "date": "datetime",
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

    # ,id,title,publication,author,date,year,month,url,content
    print(df.describe())
    print(df["id"].dtypes)
    return df[
        ["id", "title", "publication", "author", "date", "year", "month", "content"]
    ]


df = pd.concat([read_and_describe_csv(f) for f in glob.glob("articles*.csv")])

print("Combined articles")
print(df.describe())
print(df.columns)

df.to_csv("combined_articles.csv")
df.to_msgpack("./combined_articles.pack")


# pd.read_csv(
#     "./%s.csv" % filename,
#     index_col=[0],
#     header=1,
#     names=[
#         "id",
#         "title",
#         "publication",
#         "author",
#         "date",
#         "year",
#         "month",
#         "url",
#         "content",
#     ],
# )


# df1 = pd.read_csv("./articles1.csv")
# df2 = pd.read_csv("./articles2.csv")
# df3 = pd.read_csv("./articles3.csv")

# print(df1.columns)
# print(df1.describe())

# print(df2.columns)
# print(df2.describe())

# print(df3.columns)
# print(df3.describe())

# df = pd.concat(
#     map(pd.read_csv, glob.glob(os.path.join("", "articles*.csv"))), sort=False
# )
# df = pd.read_csv("./articles1.csv")

# ['Unnamed: 0', 'id', 'title', 'publication', 'author', 'date', 'year',
#        'month', 'url', 'content', 'Unnamed: 0.1'],
#       dtype='object'

# df = pd.concat(
#     [pd.read_csv(f) for f in glob.glob("articles*.csv")][
#         ["id", "title", "publication", "author", "date", "year", "month", "content"]
#     ],
#     # ignore_index=True,
# )

# print(df.columns)

# print("rows with bad index should be culled ---")
# df["id"] = df["id"].apply(pd.to_numeric, errors="coerce")
# print(df.describe())
# new_df = df[df.notna()].dropna(subset=["id"])
# print(new_df.describe())

# new_df.to_csv("articles_combined.csv")
# new_df.to_msgpack("./articles_combined.pack")

