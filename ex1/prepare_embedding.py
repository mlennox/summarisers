class PrepareEmbeddings:
    force_load_glove = False
    vocab_size = 40000
    embedding_dimension = 100
    glove_threshold = 0.5

    def run(self):
        # set to True to always reload GloVe weigths from raw file

        model_weights, model_index = load_embeddings(
            embedding_dimension, force_load_glove
        )
        df = load_data("combined_articles")

        vocabulary, vocabulary_count = build_vocabulary(
            df["title"] + " " + df["content"]
        )

        print(
            "+ + + + + + ++ + + + + - - - - - - ",
            type(model_index),
            type(model_index["the"]),
            type(model_weights),
            type(model_weights[model_index["the"]]),
        )
