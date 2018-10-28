from numpy import reshape, array, argmin
from scipy import spatial

class VocabularyMatrix(object):
      def create(
          self,
          vocabulary_size,
          vocabulary_list,
          embedding_dimension,
          model_weights,
          model_index,
          glove_threshold
      ):
        """[summary]
        
        Arguments:
          vocabulary_size {int} -- the chosen vocab size
          vocabulary_list {list} -- the list of all the words in the source dataset
          embedding_dimension {int} -- the dimension of the embedding vectors
          model_weights {dictionary<list>} -- entire embedding vectors
          model_index {dictionary} -- words of embedding as key and index as value
          glove_threshold {float} -- how close am 'outside' word needs to be to be deemed a match 
        
        Returns:
          [array] -- our vocabulary matrix we'll use for encoding
        """
        
        vocabulary_matrix_list, unmatched_words = self.inside_words(vocabulary_size, vocabulary_list, model_weights, model_index)

        return vocabulary_matrix


      def inside_words(self, vocabulary_size, vocabulary_list, model_weights, model_index):
        unmatched_words = {}
        vocabulary_matrix_list = []
        model_index_keys = list(model_index.keys())

        for index in range(vocabulary_size):
          # check if the current word exists in the model
          word = vocabulary_list[index]
          if word in model_index_keys:
              word_index = model_index[word]
              # we extend the list by the vocabulary index and the associated embedding weights
              # the vocabulary index is not the same as the embedding index
              vocabulary_matrix_list.extend([index, model_weights[word_index]])
          else:
            unmatched_words[index] = word
        return vocabulary_matrix_list, unmatched_words


      # def outside_words(self):




# 
# 

    def match_outside_words(
        self,
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
        self,
        vocab_size,
        vocab,
        embedding_dimension,
        model_weights,
        model_index,
        glove_threshold,
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

        self.match_outside_words(
            glove_threshold,
            vocab,
            vocabulary_dict,
            vocabulary_matrix,
            words_outside,
            model_index,
            model_weights,
        )
