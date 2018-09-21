import pandas as pd

import glob, os

df = pd.concat(map(pd.read_csv, glob.glob(os.path.join("", "articles*.csv"))))

df.to_csv("articles_combined.csv")

