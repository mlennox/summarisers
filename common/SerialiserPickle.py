import pickle
from traceback import print_exc


class SerialiserPickle:
    # filename_template = "./datasets/glove.6B/{0}.{1}.pkl"

    def load(self, filename):
        print("Loading data '{0}' from pickle".format(filename))
        try:
            with open(filename, "rb") as infile:
                packed = pickle.load(infile)
            return packed
        except Exception:
            print("Could not load '{0}'".format(filename))
            print_exc()
            exit()

    def save(self, data, filename):
        print("Saving data '{0}' to pickle".format(filename))
        try:
            with open(filename, "wb") as outfile:
                pickle.dump(data, outfile)
        except Exception:
            print("Could not save '{0}'".format(filename))
            print_exc()
            exit()

