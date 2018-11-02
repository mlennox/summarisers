from numpy import reshape, array, argmin, nditer, concatenate
from scipy import spatial
from common import SerialiserPickle


class VocabularyMatrix(object):
    def __init(self):
        self.serialiser = SerialiserPickle.SerialiserPickle()

    def create(
        self,
        vocabulary_size,
        vocabulary_list,
        embedding_dimension,
        model_weights,
        model_index,
        outside_threshold,
    ):
        """
        Find the embedding weights for top words in vocabulary up to the 'vocabulary_size' 
        create a matrix with vocabulary word index and weight as embedding_dimension + 1 vector
        take the words outside the top words and find embedding weights
        map outside words to existing vocabulary if their cosine distance is less than 'outside_threshold'

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
        print("Building the vocabulary matrix")

        vocabulary_matrix, vocabulary_outside_matrix, unmatched_words = self.build_matrix(
            vocabulary_size,
            vocabulary_list,
            model_weights,
            model_index,
            embedding_dimension,
        )

        nearly_inside_words = self.map_outside_words(
            vocabulary_matrix,
            vocabulary_outside_matrix,
            outside_threshold,
            embedding_dimension,
        )

        final_vocab_matrix = concatenate(vocabulary_matrix, nearly_inside_words, axis=0)

        self.serialise(final_vocab_matrix)

        return final_vocab_matrix

    def build_matrix(
        self,
        vocabulary_size,
        vocabulary_list,
        model_weights,
        model_index,
        embedding_dimension,
    ):
        """
        find the embedding weights for the 'inside' and 'outside' words
        
        Arguments:
          vocabulary_size {int} -- [description]
          vocabulary_list {list} -- [description]
          model_weights {dictionary<list>} -- [description]
          model_index {dictionary} -- [description]
        
        Returns:
          tuple(ndarray, ndarray, dictionary) -- the inside and outside vocabulary matrices and the unmatched words
        """
        vocabulary_list_length = len(vocabulary_list)
        max_words_limit = vocabulary_size + 1000
        vocabulary_matrix, unmatched_words = self.weight_words(
            vocabulary_list,
            0,
            vocabulary_size,
            model_weights,
            model_index,
            embedding_dimension,
        )
        vocabulary_outside_matrix, unmatched_outside_words = self.weight_words(
            vocabulary_list,
            vocabulary_size,
            max_words_limit,
            model_weights,
            model_index,
            embedding_dimension,
        )

        print("Vocabulary_matrix shape : {0}".format(vocabulary_matrix.shape))
        print(
            "Vocabulary_outside_matrix shape : {0}".format(
                vocabulary_outside_matrix.shape
            )
        )
        print("Unmatched inside word count : {0}".format(len(unmatched_words)))
        print("Unmatched outside word count : {0}".format(len(unmatched_outside_words)))
        print(
            "unmatched inside words - = - = - = - = - = - = - = - = - = - = - = - =",
            unmatched_words,
        )
        print(
            "unmatched outside words - = - = - = - = - = - = - = - = - = - = - = - =",
            unmatched_outside_words,
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
        embedding_dimension,
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

        for index in range(vocabulary_index_start, vocabulary_index_end):
            # check if the current word exists in the model
            word = vocabulary_list[index]
            # if index % 100 == 0:
            #     print('Checking "{0}" (index {1})'.format(word, index))
            if word in model_index_keys:
                word_index = model_index[word]
                embedding = model_weights[word_index]
                # we extend the list by the vocabulary index and the associated embedding weights
                # the vocabulary index is not the same as the embedding index
                vocabulary_matrix_list.append(index)
                vocabulary_matrix_list.extend(embedding)
            else:
                unmatched_words[index] = word
        vocabulary_matrix = reshape(
            vocabulary_matrix_list, (-1, embedding_dimension + 1)
        )
        print("= = = = = = = = = = = = =")
        print(
            "matrix built - embedding_dimension: {0} :: vocabulary_matrix shape : {1}".format(
                embedding_dimension, vocabulary_matrix.shape
            )
        )
        return vocabulary_matrix, unmatched_words

    def map_outside_words(
        self,
        vocabulary_matrix,
        vocabulary_outside_matrix,
        outside_threshold,
        embedding_dimension,
    ):
        """[summary]
        
        Arguments:
          vocabulary_matrix {[type]} -- [description]
          vocabulary_outside_matrix {[type]} -- [description]
          outside_threshold {[type]} -- [description]
        
        Returns:
          [type] -- [description]
        """
        print("Matching the outside words - - - - - - - - - - - - - - - - -")
        max_check = 1000
        nearly_inside_matrix_list = []
        vocabulary_matrix_without_index = vocabulary_matrix[:, 1:]
        vocabulary_outside_matrix_without_index = vocabulary_outside_matrix[:, 1:]
        distance_matrix_shape = (
            vocabulary_matrix_without_index.shape[0],
            vocabulary_outside_matrix_without_index.shape[0],
        )
        print(
            "shape of Vocabulary '{0}' and outside vocabulary '{1}' matrix".format(
                vocabulary_matrix_without_index.shape,
                vocabulary_outside_matrix_without_index.shape,
            )
        )
        distance_matrix = spatial.distance.cdist(
            vocabulary_matrix[:, 1:], vocabulary_outside_matrix[:, 1:], "cosine"
        ).reshape(-1)
        print("distance matrix shape", distance_matrix.shape)
        reshaped = reshape(distance_matrix, distance_matrix_shape)
        print("reshaped", reshaped.shape)

        # # note: need to figure out how to vectorise this
        # for word_and_weight in nditer(
        #     vocabulary_outside_matrix, flags=["external_loop", "buffered"], order="F"
        # ):
        #     # need to trim first element from word and weight as we only want the glove embedding
        #     word_and_weight_vector_without_index = array(word_and_weight[1:]).reshape(
        #         1, -1
        #     )
        #     print(
        #         "word_and_weight_vector_without_index shape",
        #         word_and_weight_vector_without_index.shape,
        #     )
        #     distance_matrix = spatial.distance.cdist(
        #         vocabulary_matrix_without_index,
        #         word_and_weight_vector_without_index,
        #         "cosine",
        #     ).reshape(-1)
        #     min_index = argmin(distance_matrix)
        #     if distance_matrix[min_index] <= outside_threshold:
        #         nearly_inside_matrix_list.append(vocabulary_matrix[min_index])

        # print(
        #     "There were {0} outside words that were a close enough match to be mapped to words in the vocabulary matrix".format(
        #         len(nearly_inside_matrix_list)
        #     )
        # )
        return nearly_inside_matrix_list

    def serialise(self, vocabulary_matrix):
        self.serialiser.save(vocabulary_matrix, "./datasets/vocabulary_matrix.pkl")
