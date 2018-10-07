from collections import Counter
from itertools import chain
from pandas import read_msgpack
from numpy import array
from traceback import print_exc
from msgpack import pack
from msgpack_numpy import encode


def load_data(dataset_name):
    print("Loading data")
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


def load_embeddings(embedding_dimension):
    print("loading GloVe embeddings of dimension : %s" % embedding_dimension)
    try:
        model = {}
        with open(
            "../datasets/glove.6B/glove.6B.%sd.txt" % embedding_dimension, "r"
        ) as glove_file:
            for line in glove_file:
                splitLine = line.split()
                word = splitLine[0]
                embedding = array([float(val) for val in splitLine[1:]])
                model[word] = embedding
            save_embeddings(model)
            return model
    except Exception:
        print("some issue loading the file or processing the weights")
        print_exc()
        exit()


def save_embeddings(model):
    print("Saving embedding model in msgpack format")
    try:
        with open("../datasets/glove.6B/model.pack", "wb") as outfile:
            # x_enc = msgpack.packb(x, default=m.encode)
            # x_rec = msgpack.unpackb(x_enc, object_hook=m.decode)
            pack(model, outfile, default=encode)
    except Exception:
        print("Could not save the embeddings model")
        print_exc()
        exit()


def run():
    vocab_size = 40000
    embedding_dimension = 100
    df = load_data("combined_articles")
    vocab, vocabulary_count = build_vocabulary(
        df["title"].apply(remove_quotes) + df["content"].apply(remove_quotes)
    )
    model = load_embeddings(embedding_dimension)


run()
