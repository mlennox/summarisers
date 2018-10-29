from numpy import reshape, array, argmin, nditer
from scipy import spatial


class VocabularyMatrix(object):
    def create(
        self,
        vocabulary_size,
        vocabulary_list,
        embedding_dimension,
        model_weights,
        model_index,
        outside_threshold,
    ):
        """[summary]
        
        Arguments:
          vocabulary_size {int} -- the chosen vocab size
          vocabulary_list {list} -- the list of all the words in the source dataset
          embedding_dimension {int} -- the dimension of the embedding vectors
          model_weights {dictionary<list>} -- entire embedding vectors
          model_index {dictionary} -- words of embedding as key and index as value
          outside_threshold {float} -- how close (in cos distance) an 'outside' word needs to be to be deemed a match 
        
        Returns:
          [array] -- our vocabulary matrix we'll use for encoding
        """

        vocabulary_matrix, vocabulary_outside_matrix, unmatched_words = self.inside_words(
            vocabulary_size, vocabulary_list, model_weights, model_index
        )

        nearly_inside_words = self.map_outside_words(
            vocabulary_matrix, vocabulary_outside_matrix, outside_threshold
        )

        return vocabulary_matrix

    def inside_words(
        self, vocabulary_size, vocabulary_list, model_weights, model_index
    ):
        """[summary]
        
        Arguments:
          vocabulary_size {[type]} -- [description]
          vocabulary_list {[type]} -- [description]
          model_weights {[type]} -- [description]
          model_index {[type]} -- [description]
        
        Returns:
          [type] -- [description]
        """

        vocabulary_matrix, unmatched_words = self.weight_words(
            vocabulary_list, 0, vocabulary_size, model_weights, model_index
        )
        vocabulary_outside_matrix, unmatched_outside_words = self.weight_words(
            vocabulary_list,
            vocabulary_size,
            len(vocabulary_list),
            model_weights,
            model_index,
        )

        return (
            vocabulary_matrix,
            vocabulary_outside_matrix,
            {**unmatched_words, **unmatched_outside_words},
        )

    def weight_words(
        self,
        vocabulary_list,
        vocabulary_index_start,
        vocabulary_index_end,
        model_weights,
        model_index,
    ):
        """[summary]
        
        Arguments:
          vocabulary_list {[type]} -- [description]
          vocabulary_index_start {[type]} -- [description]
          vocabulary_index_end {[type]} -- [description]
          model_weights {[type]} -- [description]
          model_index {[type]} -- [description]
        
        Returns:
          [type] -- [description]
        """

        unmatched_words = {}
        vocabulary_matrix_list = []
        model_index_keys = list(model_index.keys())
        embedding_dimension = 0

        for index in range(vocabulary_index_start, vocabulary_index_end):
            # check if the current word exists in the model
            word = vocabulary_list[index]
            if word in model_index_keys:
                word_index = model_index[word]
                embedding = model_weights[word_index]
                if embedding == 0:
                    embedding_dimension = len(embedding)
                # we extend the list by the vocabulary index and the associated embedding weights
                # the vocabulary index is not the same as the embedding index
                vocabulary_matrix_list.extend([index, embedding])
            else:
                unmatched_words[index] = word
        vocabulary_matrix = reshape(vocabulary_matrix_list, (-1, embedding_dimension))
        return vocabulary_matrix, unmatched_words

    def map_outside_words(
        self, vocabulary_matrix, vocabulary_outside_matrix, outside_threshold
    ):
        """[summary]
        
        Arguments:
          vocabulary_matrix {[type]} -- [description]
          vocabulary_outside_matrix {[type]} -- [description]
          outside_threshold {[type]} -- [description]
        
        Returns:
          [type] -- [description]
        """

        nearly_inside_matrix_list = []
        # note: need to figure out how to vectorise this
        for word_and_weight in nditer(
            vocabulary_outside_matrix, flags=["external_loop"], order="F"
        ):
            # need to trim first element from word and weight as we only want the glove embedding
            distance_matrix = spatial.distance.cdist(
                vocabulary_matrix, word_and_weight, "cosine"
            ).reshape(-1)
            min_index = argmin(distance_matrix)
            if distance_matrix[min_index] <= outside_threshold:
                nearly_inside_matrix_list.append(vocabulary_matrix[min_index])
        print(
            "There were {0} outside words that were a close enough match to be mapped to words in the vocabulary matrix".format(
                len(nearly_inside_matrix_list)
            )
        )
        return nearly_inside_matrix_list

