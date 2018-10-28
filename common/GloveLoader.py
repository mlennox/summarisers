import pandas as pd
import numpy as np
from json import loads
from common import SerialiserPickle
from os import path
from traceback import print_exc


class GloveLoader(object):
    def __init__(self):
        self.serialiser = SerialiserPickle.SerialiserPickle()

    # this is relative to the root folder, not the script location...
    glovepath = "./datasets/glove.6B/"

    def load(self, embedding_dimension, use_cache):
        """
        Loads the raw glove weights and returns 
        
        Arguments:
            embedding_dimension {int} -- the dimension of the word vectors
            use_cache {boolean} -- if set to true, and the model has been loaded and saved, we will load the serialised version
        
        Returns:
            [tuple(dictionary<list>, dictionary)] -- a tuple containing a dictionary with the word index as key and a list (matching embedding_dimension in length) of the word weights
        """
        print("loading GloVe embeddings of dimension : %s" % embedding_dimension)
        try:
            model_weights = {}
            model_index = {}
            model_exists = path.isfile(
                "{0}model_weights.{1}.pkl".format(self.glovepath, embedding_dimension)
            )
            if model_exists and use_cache == True:
                print("Using cache - Loading existing serialised models")
                (model_weights, model_index) = self.deserialise(embedding_dimension)
            else:
                (model_weights, model_index) = self.load_vectors(embedding_dimension)
                # now cache the serialised models
                self.serialise(embedding_dimension, model_weights, model_index)
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
            "{0}glove.6B.{1}d.txt".format(self.glovepath, embedding_dimension), "r"
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
        """Serialise the loaded Glove model to speed up loading next time
        
        Arguments:
            embedding_dimension {int} -- the required dimension of the vector
            model_weights {dictionary<list>} -- a dictionary of the weight lists (with length embedding_dimension) where the key is the word index
            model_index {dictionary} -- a dictionary of the word as key and index as value
        """
        self.serialiser.save(model_weights, embedding_dimension, "model_weights")
        self.serialiser.save(model_index, embedding_dimension, "model_index")

    def deserialise(self, embedding_dimension):
        """deserialise the model
        
        Arguments:
            embedding_dimension {int} -- dimension of the model
        
        Returns:
            [tuple(dictionary<list>, dictionary)] -- a tuple containing a dictionary with the word index as key and a list (matching embedding_dimension in length) of the word weights
        """
        model_weights = self.serialiser.load(embedding_dimension, "model_weights")
        model_index = self.serialiser.load(embedding_dimension, "model_index")
        return (model_weights, model_index)

