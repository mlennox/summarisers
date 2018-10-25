from common import DataLoader, GloveLoader


class PrepareEmbeddings:
    def __init__(self):
        self.gloveloader = GloveLoader.GloveLoader()

    force_load_glove = False
    vocab_size = 40000
    embedding_dimension = 100
    glove_threshold = 0.5

    def run(self):
        # set to True to always reload GloVe weights from raw file

        model_weights, model_index = self.gloveloader.load(
            self.embedding_dimension, self.force_load_glove
        )
        # df = load_data("combined_articles")

        # vocabulary, vocabulary_count = build_vocabulary(
        #     df["title"] + " " + df["content"]
        # )

        # print(
        #     "+ + + + + + ++ + + + + - - - - - - ",
        #     type(model_index),
        #     type(model_index["the"]),
        #     type(model_weights),
        #     type(model_weights[model_index["the"]]),
        # )
