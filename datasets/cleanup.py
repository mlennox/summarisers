import pandas as pd
import numpy as np
from collections import Counter
from itertools import chain

df = pd.read_msgpack("./combined_articles.pack")
# df = pd.read_csv("./combined_articles.csv")
print(df.describe())

dated = df[df["date"].apply(lambda x: type(x) != str)]
print("-- - - - - -", dated.describe())

#  remove the bad rows
df = df[df["title"].apply(lambda x: type(x) == str)]
df = df[df["date"].apply(lambda x: type(x) != str)]
print(df.describe())

# substitute smart quotes for straight
df["title"] = df["title"].apply(
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

df["content"] = df["content"].apply(
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

# print("- - - - - - TITLE")
# wrong = df[df["title"].apply(lambda x: type(x) != str)]
# print(wrong.describe())
# print(wrong.head(10))

# print("- - - - - - CONTENT")
# wrong = df[df["content"].apply(lambda x: type(x) != str)]
# print(wrong.describe())
# print(wrong.head(10))

# print("- - - - - - ID")
# wrong = df[df["id"].apply(lambda x: type(x) == str)]
# print(wrong.describe())
# print(wrong.head(10))

# df[df['A'].apply(lambda x: type(x)==str)]


# def build_vocabulary(word_list):
#     where = 1
#     for txt in word_list:
#         where = where + 1
#         print("- - - - -", where, type(txt))
#     # vocabcount = Counter(word for txt in word_list for word in txt.split())
#     # return vocabcount


# vocabulary_count = build_vocabulary(df["title"])  # +df['content'])
# print(vocabulary_count)


#     vocab = map(lambda x: x[0], sorted(vocabcount.items(), key=lambda x: -x[1]))
#     return vocab, vocabcount


# print("loading the dataframe back from the file")
# df = pd.read_msgpack("./tester.pack")
# print(df.describe())

# print("rows with bad index should be culled ---")
# coerced = df.apply(pd.to_numeric, errors="coerce").notna()
# print(coerced.describe())
# new_df = coerced.dropna()
# print(new_df.describe())
# new_df.to_msgpack("./articles_combined")


# def read_and_describe_csv(filename):
#     df = pd.read_csv(
#         # f,
#         "./%s.csv" % filename,
#         # index_col="id",
#         date_parser=pd.to_datetime,
#         dtype={
#             "id": np.int64,
#             "title": np.str,
#             "publication": np.str,
#             "author": np.str,
#             "content": np.str,
#             # "date": "datetime",
#             "year": np.float,
#             "month": np.float,
#         },
#         header=0,
#         names=[
#             "ignore",
#             "id",
#             "title",
#             "publication",
#             "author",
#             "date",
#             "year",
#             "month",
#             "url",
#             "content",
#         ],
#     )

#     # ,id,title,publication,author,date,year,month,url,content
#     print(df.describe())
#     print(df["id"].dtypes)
#     return df[
#         ["id", "title", "publication", "author", "date", "year", "month", "content"]
#     ]


# df1 = read_and_describe_csv("articles1")

# df2 = read_and_describe_csv("articles2")

# df3 = read_and_describe_csv("articles3")

# df_combined = pd.concat([df1, df2, df3])

# print(df_combined.describe())
# print(df_combined.columns)
# print(df_combined["id"].dtypes)

