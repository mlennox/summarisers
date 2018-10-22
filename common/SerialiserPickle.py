import pickle
from traceback import print_exc


class SerialiserPickle:
    def load(self, filename):
        print("Loading data '%s' from pickle" % filename)
        try:
            with open("../datasets/glove.6B/%s.pkl" % filename, "rb") as infile:
                packed = pickle.load(infile)
            return packed
        except Exception:
            print("Could not load '%s'" % filename)
            print_exc()
            exit()

    def save(self, data, filename):
        print("Loading data '%s' from pickle" % filename)
        try:
            with open("../datasets/glove.6B/%s.pkl" % filename, "wb") as outfile:
                pickle.dump(data, outfile)
        except Exception:
            print("Could not save '%s'" % filename)
            print_exc()
            exit()

