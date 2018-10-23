import pickle
from traceback import print_exc


class SerialiserPickle:
    def load(self, embedding_dimension, filename):
        print("Loading data '%s' from pickle" % filename)
        try:
            with open(
                "../datasets/glove.6B/{0}.{1}.pkl".format(
                    filename, embedding_dimension
                ),
                "rb",
            ) as infile:
                packed = pickle.load(infile)
            return packed
        except Exception:
            print("Could not load '{0}.{1}'".format(filename, embedding_dimension))
            print_exc()
            exit()

    def save(self, data, embedding_dimension, filename):
        print("Loading data '%s' from pickle" % filename)
        try:
            with open(
                "../datasets/glove.6B/{0}.{1}.pkl".format(
                    filename, embedding_dimension
                ),
                "wb",
            ) as outfile:
                pickle.dump(data, outfile)
        except Exception:
            print("Could not save '{0}.{1}'".format(filename, embedding_dimension))
            print_exc()
            exit()

