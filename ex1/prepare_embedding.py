from collections import Counter
from itertools import chain
from pandas import read_msgpack
from numpy import array, empty, random
from traceback import print_exc
from msgpack import pack, unpack, packb, unpackb
from os import path
import msgpack_numpy as m

m.patch()  # patch msgpack to use msgpack_numpy encoding/decoding


def load_data(dataset_name):
    print("Loading news dataset from msgpack")
    try:
        return read_msgpack("../datasets/%s.pack" % dataset_name)
    except Exception:
        print(
            "Some problem loading the articles_combined.pack - you need to unzip your csv and run combine.py first!"
        )
        exit()


remove_quotes = lambda x: x.replace('"', "")


def build_vocabulary(word_list):
    print("building vocabulary")
    vocabcount = Counter(word for txt in word_list for word in txt.split())
    vocab = map(lambda x: x[0], sorted(vocabcount.items(), key=lambda x: -x[1]))
    return list(vocab), vocabcount


def load_embeddings(embedding_dimension, force_load_glove):
    print("loading GloVe embeddings of dimension : %s" % embedding_dimension)
    try:
        model_weights = {}
        model_index = {}
        model_exists = path.isfile("../datasets/glove.6B/model_weights.pack")
        if model_exists and force_load_glove != True:
            print("Loading existing msgpacked models")
            model_weights = load_from_msgpack("model_weights")
            model_index = load_from_msgpack("model_index")
        else:
            print("Loading raw GloVe embeddings and creating msgpack")
            with open(
                "../datasets/glove.6B/glove.6B.%sd.txt" % embedding_dimension, "r"
            ) as glove_file:
                index = 1
                for line in glove_file:
                    splitLine = line.split()
                    word = splitLine[0]
                    embedding = array([float(val) for val in splitLine[1:]])
                    model_weights[index] = embedding
                    model_index[word] = index
                    index += 1
                save_to_msgpack(model_weights, "model_weights")
                save_to_msgpack(model_index, "model_index")
        return model_weights, model_index
    except Exception:
        print("some issue loading the file or processing the weights")
        print_exc()
        exit()


def load_from_msgpack(filename):
    print("Loading data '%s' from msgpack format" % filename)
    try:
        with open("../datasets/glove.6B/%s.pack" % filename, "rb") as infile:
            packed = infile.read()
        return unpackb(packed)
    except Exception:
        print("Could not load '%s'" % filename)
        print_exc()
        exit()


def save_to_msgpack(data, filename):
    print("Saving data '%s' in msgpack format" % filename)
    try:
        packed = packb(data)
        with open("../datasets/glove.6B/%s.pack" % filename, "wb") as outfile:
            outfile.write(packed)
    except Exception:
        print("Could not save '%s'" % filename)
        print_exc()
        exit()


# copy any matching word embeddings from GloVe to our vocabulary matrix
def create_vocab_matrix(
    vocab_size, vocab, embedding_dimension, model_weights, model_index, glove_threshold
):
    print("Creating the vocabulary matrix")
    # matrix_shape = (vocab_size, embedding_dimension)
    # scale = 1  # what to do here...?
    vocabulary_matrix = {}
    # random.uniform(low= -scale, high=scale, size=matrix_shape)
    model_index_keys = model_index.keys()

    for index in range(vocab_size):
        # check if the current word exists in the model
        word = vocab[index]
        if word in model_index_keys:
            word_index = model_index[word]
            vocabulary_matrix[word_index] = model_weights[word_index]
    print("Entries in vocabulary matrix", len(vocabulary_matrix))

    # now get nearest match embeddings for words that are not in the GloVe embeddings


def run():
    force_load_glove = False
    vocab_size = 40000
    embedding_dimension = 100
    glove_threshold = 0.5
    model_weights, model_index = load_embeddings(embedding_dimension, force_load_glove)
    df = load_data("combined_articles")
    vocab, vocabulary_count = build_vocabulary(
        df["title"].apply(remove_quotes) + df["content"].apply(remove_quotes)
    )
    create_vocab_matrix(
        vocab_size,
        vocab,
        embedding_dimension,
        model_weights,
        model_index,
        glove_threshold,
    )


run()
