import pandas as pd
from json import loads


class ArticleLoader:
    filename = "./datasets/signalmedia-1m.jsonl"

    def load(self):
        """
        Load the JSONL (json object per line) file into a dataframe        
        
        Returns:
            [DataFrame] -- a pandas dataframe of the JSON data
        """

        df = None
        try:
            print("loading the JSONL")
            json_list = []
            with open(self.filename, "r") as json_file:
                for line in json_file:
                    json_list.append(loads(line))
            print("Converting to dataframe")
            df = pd.io.json.json_normalize(json_list)
        except Exception as e:
            print("Some problem loading the '{0}'".format(self.filename))
            print(e)
            exit()

        # print("Columns : ", df.columns)
        # print("Working on describing the data")
        # print(df.describe())

        return df
