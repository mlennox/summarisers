from common import ArticleLoader, GloveLoader, Vocabulary


class PrepareEmbeddings:
    def __init__(self):
        self.gloveloader = GloveLoader.GloveLoader()
        self.articleloader = ArticleLoader.ArticleLoader()
        self.vocabulary = Vocabulary.Vocabulary()

    use_cache = True
    vocab_size = 40000
    embedding_dimension = 100
    glove_threshold = 0.5

    def run(self):
        # set to True to always reload GloVe weights from raw file

        model_weights, model_index = self.gloveloader.load(
            self.embedding_dimension, self.use_cache
        )
        article_df = self.articleloader.load()

        vocabulary, vocabulary_count = self.vocabulary.build_vocabulary(
            article_df, self.use_cache
        )


prepare = PrepareEmbeddings()
prepare.run()
