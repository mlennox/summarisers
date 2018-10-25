import pandas as pd
import numpy as np
from json import loads
from common import SerialiserPickle
from os import path
from traceback import print_exc


class GloveLoader(object):
    def __init__(self):
        self.serialiser = SerialiserPickle.SerialiserPickle()

    glovepath = "../datasets/glove.6B/"

    def load(self, embedding_dimension, force_load_glove):
        print("loading GloVe embeddings of dimension : %s" % embedding_dimension)

        try:
            model_weights = {}
            model_index = {}
            model_exists = path.isfile(
                "../datasets/glove.6B/model_weights.{0}.pkl".format(embedding_dimension)
            )
            if model_exists and force_load_glove != True:
                print("Loading existing msgpacked models")
                (model_weights, model_index) = self.deserialise(embedding_dimension)
            else:
                (model_weights, model_index) = self.load_vectors(embedding_dimension)
                self.serialise(model_weights, embedding_dimension, "model_weights")
                self.serialise(model_index, embedding_dimension, "model_index")
            return model_weights, model_index
        except Exception as e:
            print("some issue loading the file or processing the weights", e)
            print_exc()
            exit()

    def load_vectors(self, embedding_dimension):
        print("Loading raw GloVe embeddings")
        model_weights = {}
        model_index = {}
        with open(
            "{0}glove.6B.{1}.txt".format(self.glovepath, embedding_dimension), "r"
        ) as glove_file:
            index = 1
            for line in glove_file:
                splitLine = line.split()
                word = splitLine[0]
                embedding = [float(val) for val in splitLine[1:]]
                model_weights[index] = embedding
                model_index[word] = index
                index += 1
        return model_weights, model_index

    def serialise(self, embedding_dimension, model_weights, model_index):
        """
        Serialise the loaded Glove model to speed up loading next time
        """
        self.serialiser.save(model_weights, embedding_dimension, "model_weights")
        self.serialiser.save(model_index, embedding_dimension, "model_index")

    def deserialise(self, embedding_dimension):
        model_weights = self.serialiser.load(embedding_dimension, "model_weights")
        model_index = self.serialiser.load(embedding_dimension, "model_index")
        return (model_weights, model_index)

