from collections import Counter
from itertools import chain
from pandas import read_msgpack
from numpy import array, empty, random, ndarray, reshape, argmin
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


whittle_to_words = lambda x: re.sub(r"[^\w\s'\b]", " ", x).lower()


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
        print("* * * ** * ** *")
        print("source dataset has '%s' unique words" % len(vocab))
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


def match_outside_words(
    glove_threshold,
    vocab,
    vocabulary_matrix,
    vocabulary_dict,
    words_outside,
    model_index,
    model_weights,
):
    """
    Iterate through the outside words
    find the embedding vector from the GloVe model
    compare the resulting vector with all vectors of vocabulary matrix
    assign the outside word to a vocabulary matrix if they are similar enough
    """
    reject_words = []
    vocabulary_dict_keys = list(vocabulary_dict.keys())
    for outside_word_index, outside_word in words_outside.items():
        if outside_word in model_index:
            outside_word_glove_index = model_index[outside_word]
            outside_word_glove_vector = array(
                model_weights[outside_word_glove_index]
            ).reshape(1, -1)

            distance_matrix = spatial.distance.cdist(
                vocabulary_matrix, outside_word_glove_vector, "cosine"
            ).reshape(-1)
            min_index = argmin(distance_matrix)

            if distance_matrix[min_index] <= glove_threshold:
                print(
                    "word: {0} -- glove index: {1} -- vocab index: {2} -- distance score: {3}".format(
                        outside_word,
                        model_index[outside_word],
                        vocabulary_dict_keys[min_index],
                        distance_matrix[min_index],
                    )
                )
                print(
                    "The outside word '{0}' is close enough to '{1}'".format(
                        outside_word, vocab[vocabulary_dict_keys[min_index]]
                    )
                )

        else:
            print('The word "%s" was not in the GloVe list' % outside_word)
            reject_words.append(outside_word)

    print("Reject count %s" % len(reject_words))
    print("Outside count %s" % len(words_outside))


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
    # print("model index keys - - - -- ", model_index_keys[1:50])
    # print("vocabulary - - - -- ", vocab[1:50])

    for index in range(vocab_size):
        # check if the current word exists in the model
        word = vocab[index]
        if word in model_index_keys:
            word_index = model_index[word]
            # vocabulary_dict[word_index] = model_weights[word_index]
            # is this as performant as possible?
            vocabulary_matrix_list.extend(model_weights[word_index])
        else:
            # now we split by single quote as many outside words contain or are pre/suffixed by them
            for split_word in word.split("'"):
                if split_word not in vocab and len(split_word) > 0:
                    print("NOT IN VOCAB", split_word)
                    words_outside[len(words_outside)] = split_word

    print("Entries in vocabulary matrix", len(vocabulary_dict))
    print("Words outside the vocabulary matrix", len(words_outside))
    # print("vocab matrix list - first row", vocabulary_matrix_list[0:202])
    vocabulary_matrix = reshape(vocabulary_matrix_list, (-1, embedding_dimension))
    # print("vocab matrix - first two rows", vocabulary_matrix[0:2])

    match_outside_words(
        glove_threshold,
        vocab,
        vocabulary_dict,
        vocabulary_matrix,
        words_outside,
        model_index,
        model_weights,
    )


def build_vocab_matrix(
    vocab_size, vocab, embedding_dimension, model_weights, model_index, glove_threshold
):
    """

    
    Arguments:
        vocab_size {int} -- sets the size limit of the vocabulary 
        vocab {list<str>} -- list of vocabulary words (<= vocab_size)
        embedding_dimension {int} -- the size of the GloVe embedding vector
        model_weights {[dict<word,]} -- [description]
        model_index {[type]} -- [description]
        glove_threshold {[type]} -- [description]
    """


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


def run():
    # set to True to always reload GloVe weigths from raw file
    force_load_glove = False
    vocab_size = 40000
    embedding_dimension = 100
    glove_threshold = 0.5
    model_weights, model_index = load_embeddings(embedding_dimension, force_load_glove)
    df = load_data("combined_articles")

    vocabulary, vocabulary_count = build_vocabulary(df["title"] + " " + df["content"])

    print(
        "+ + + + + + ++ + + + + - - - - - - ",
        type(model_index),
        type(model_index["the"]),
        type(model_weights),
        type(model_weights[model_index["the"]]),
    )
    # create_vocab_matrix(
    #     vocab_size,
    #     vocab,
    #     embedding_dimension,
    #     model_weights,
    #     model_index,
    #     glove_threshold,
    # )


run()
