import pandas as pd
import numpy as np
from json import loads


class GloveLoader:
    glovepath = "../datasets/glove.6B/"

    def load(self):
        """
        Load the Glove embeddings
        
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

        print("Columns : ", df.columns)
        print("Working on describing the data")
        print(df.describe())

        return df

    def serialise(self, model_weights, model_index):
        """
      Serialise the loaded Glove model to speed up loading next time
      """
      save_pickle(model_weights, "model_weights")
      save_pickle(model_index, "model_index")


def load_embeddings(embedding_dimension, force_load_glove):
    print("loading GloVe embeddings of dimension : %s" % embedding_dimension)
    try:
        model_weights = {}
        model_index = {}
        model_exists = path.isfile("../datasets/glove.6B/model_weights.pkl")
        if model_exists and force_load_glove != True:
            print("Loading existing msgpacked models")
            model_weights = load_pickle("model_weights")
            model_index = load_pickle("model_index")
        else:
            print("Loading raw GloVe embeddings and creating msgpack")
            with open(
                "../datasets/glove.6B/glove.6B.%sd.txt" % embedding_dimension, "r"
            ) as glove_file:
                index = 1
                for line in glove_file:
                    splitLine = line.split()
                    word = splitLine[0]
                    embedding = [float(val) for val in splitLine[1:]]
                    model_weights[index] = embedding
                    model_index[word] = index
                    index += 1
                save_pickle(model_weights, "model_weights")
                save_pickle(model_index, "model_index")
        return model_weights, model_index
    except Exception:
        print("some issue loading the file or processing the weights")
        print_exc()
        exit()
