from collections import Counter
from itertools import chain
from pandas import read_msgpack
from numpy import array, empty, random, ndarray, reshape
from traceback import print_exc
import pickle

# from msgpack import packb, unpackb, ExtType
from os import path
import re
from scipy import spatial

# import msgpack_numpy as m

# m.patch()  # patch msgpack to use msgpack_numpy encoding/decoding


def load_data(dataset_name):
    print("Loading news dataset from msgpack")
    try:
        return read_msgpack("../datasets/%s.pack" % dataset_name)
    except Exception:
        print(
            "Some problem loading the articles_combined.pack - you need to unzip your csv and run combine.py first!"
        )
        exit()


whittle_to_words = lambda x: re.sub(r"[^\w\s'\b]", " ", x)


def build_vocabulary(word_list):
    print("building vocabulary")
    vocab_exists = path.isfile("../datasets/glove.6B/vocab.pkl")
    if vocab_exists:
        print("We can load the pre-prepared vocab list")
        vocab = load_pickle("vocab")
        vocabcount = load_pickle("vocabcount")
    else:
        print("We need to generate the vocab list")
        word_list = word_list.apply(whittle_to_words)
        vocabcount = Counter(word for txt in word_list for word in txt.split())
        vocab = list(
            map(lambda x: x[0], sorted(vocabcount.items(), key=lambda x: -x[1]))
        )
        save_pickle(vocab, "vocab")
        save_pickle(vocabcount, "vocabcount")
    return vocab, vocabcount


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


def load_pickle(filename):
    print("Loading data '%s' from pickle" % filename)
    try:
        with open("../datasets/glove.6B/%s.pkl" % filename, "rb") as infile:
            packed = pickle.load(infile)
        return packed
    except Exception:
        print("Could not load '%s'" % filename)
        print_exc()
        exit()


def save_pickle(data, filename):
    print("Loading data '%s' from pickle" % filename)
    try:
        with open("../datasets/glove.6B/%s.pkl" % filename, "wb") as outfile:
            pickle.dump(data, outfile)
    except Exception:
        print("Could not save '%s'" % filename)
        print_exc()
        exit()


# def default(obj):
#     # print("= = = =")
#     # print("obj : ", obj)
#     # print("obj type: ", type(obj))
#     if isinstance(obj, ndarray):  # and obj.typecode == "d":
#         return ExtType(42, obj.tostring())
#     raise TypeError("Unknown type: %r" % (obj,))


# def ext_hook(code, data):
#     print("= = = = = =")
#     print("code : ", code)
#     print("data : ", data)
#     if code == 42:
#         a = array("d")
#         a.fromstring(data)
#         return a
#     return ExtType(code, data)


# def load_from_msgpack(filename):
#     print("Loading data '%s' from msgpack format" % filename)
#     try:
#         with open("../datasets/glove.6B/%s.pack" % filename, "rb") as infile:
#             packed = infile.read()
#         # return unpackb(
#         #     packed,
#         # )  # streaming msgpack does not work with msgpack-numpy
#         return unpackb(packed, ext_hook=ext_hook, raw=False)
#     except Exception:
#         print("Could not load '%s'" % filename)
#         print_exc()
#         exit()


# def save_to_msgpack(data, filename):
#     print("Saving data '%s' in msgpack format" % filename)
#     try:
#         packed = packb(
#             data, default=default, use_bin_type=True
#         )  # streaming msgpack does not work with msgpack-numpy
#         with open("../datasets/glove.6B/%s.pack" % filename, "wb") as outfile:
#             outfile.write(packed)
#     except Exception:
#         print("Could not save '%s'" % filename)
#         print_exc()
#         exit()


def create_vocab_matrix(
    vocab_size, vocab, embedding_dimension, model_weights, model_index, glove_threshold
):
    """
    copy any matching word embeddings from GloVe to our vocabulary matrix
    """
    print("Creating the vocabulary matrix")
    vocabulary_dict = {}
    words_outside = {}
    # we will add the word index and weights to a list which is 101 elements at a time
    # then we will reshape that into a 101x{vocab size} array/matrix
    vocabulary_matrix_list = []
    model_index_keys = list(model_index.keys())
    print("model index keys - - - -- ", model_index_keys[1:50])
    print("vocabulary - - - -- ", vocab[1:50])

    for index in range(vocab_size):
        # check if the current word exists in the model
        word = vocab[index]
        if word in model_index_keys:
            word_index = model_index[word]
            vocabulary_dict[word_index] = model_weights[word_index]
            # is this as performant as possible?
            vocabulary_matrix_list.extend([word_index] + model_weights[word_index])
        else:
            words_outside[index] = word
    print("Entries in vocabulary matrix", len(vocabulary_dict))
    print("Words outside the vocabulary matrix", len(words_outside))
    print("vocab matrix list - first row", vocabulary_matrix_list[0:202])
    vocabulary_matrix = reshape(vocabulary_matrix_list, (-1, 101))
    print("vocab matrix - first two rows", vocabulary_matrix[0:2])


#     # now get nearest match embeddings for words that are not in the GloVe embeddings
#     # convert the dict to a matrix
#     # get embedding for the word from GloVe
#     # compare to the embeddings in matrix
#     # select best - or exclude if not close match
#     for index in words_outside.keys:


# def cos_cdist(matrix, vector):
#     """
#     Compute the cosine distances between each row of matrix and vector.
#     """
#     v = vector.reshape(1, -1)
#     return spatial.distance.cdist(matrix, v, "cosine").reshape(-1)


def run():
    # set to True to always reload GloVe weigths from raw file
    force_load_glove = False
    vocab_size = 40000
    embedding_dimension = 100
    glove_threshold = 0.5
    model_weights, model_index = load_embeddings(embedding_dimension, force_load_glove)
    df = load_data("combined_articles")

    vocab, vocabulary_count = build_vocabulary(df["title"] + " " + df["content"])
    create_vocab_matrix(
        vocab_size,
        vocab,
        embedding_dimension,
        model_weights,
        model_index,
        glove_threshold,
    )


run()
