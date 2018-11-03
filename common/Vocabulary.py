import re
from common import SerialiserPickle
from os import path
from collections import Counter
from traceback import print_exc


class Vocabulary(object):
    def __init__(self):
        self.serialiser = SerialiserPickle.SerialiserPickle()

    filename_template = "./datasets/glove.6B/"

    whittle_to_words = staticmethod(lambda x: re.sub(r"[^\w\s'\-/]", " ", x))

    remove_multiple_whitespace = staticmethod(lambda x: re.sub(r"\s+", " ", x))

    straighten_single_quotes = staticmethod(lambda x: re.sub(r"[‛’‘’‵′]", "'", x))

    # straighten_double_quotes = staticmethod(lambda x: re.sub(r"[“”‶″]", '"', x))

    def make_word_list(self, df):
        df["title"] = (
            df["title"]
            .apply(self.straighten_single_quotes)
            .apply(self.whittle_to_words)
            .apply(self.remove_multiple_whitespace)
        )
        df["content"] = (
            df["content"]
            .apply(self.straighten_single_quotes)
            .apply(self.whittle_to_words)
            .apply(self.remove_multiple_whitespace)
        )
        return df["title"] + " " + df["content"]

    def process_word_list(self, word_list):
        vocabcount = Counter(
            word for txt in word_list for word in re.split(r"[\s\\\/]", txt)
        )
        vocab = list(
            map(lambda x: x[0], sorted(vocabcount.items(), key=lambda x: -x[1]))
        )
        print("Source dataset has '{0}' unique words".format(len(vocab)))
        return (vocab, vocabcount)

    def build_vocabulary(self, df, use_cache):
        print("building vocabulary")
        try:
            vocab_exists = path.isfile("{0}vocab.pkl".format(self.filename_template))
            if vocab_exists and use_cache == True:
                print("We can load the pre-prepared vocab list")
                vocab = self.deserialise("vocab")
                vocabcount = self.deserialise("vocabcount")
            else:
                print("We need to generate the vocab list")
                vocab, vocabcount = self.process_word_list(self.make_word_list(df))
                self.serialise(vocab, "vocab")
                self.serialise(vocabcount, "vocabcount")
        except Exception as e:
            print("some issue loading the file or processing the weights", e)
            print_exc()
            exit()

        return vocab, vocabcount

    def deserialise(self, model_name):
        """Load the vocabulary list and index
        
        Arguments:
          filename_slug {str} -- partial filename combined with template to create full file name
        
        Returns:
          [*] -- returns the desrialised model - can be any type...
        """
        filename = "{0}{1}.pkl".format(self.filename_template, model_name)
        print("Desrialising model from '{0}'".format(filename))
        return self.serialiser.load(filename)

    def serialise(self, model, model_name):
        """Serialise the data to file
        
        Arguments:
          model {object} -- the model data
          model_name {str} -- used to form the filename, usually choose the model variable name
        """
        filename = "{0}{1}.pkl".format(self.filename_template, model_name)
        self.serialiser.save(model, filename)

