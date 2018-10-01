# Seq2seq - LSTM and Attention

## Pre requisites

You'll need to have run `setup.sh` in the root folder - but you'll also need to have [installed Pyenv and virutalenv](http://www.webpusher.ie/2018/09/19/python-dependency-hell-no/)

## Overview of approach

We will be using a Recurrent Neural Network (RNN) to process and encode the content of an article as a sequence of words. This encoding will be passed to a second RNN that will decode the encoded sequence as a second sequence of words.

The encoder and decoder are trained at the same time using testing content with corresponding summaries. Since RNNs require input as vectors, each word of the sequence is converted to a vector using a reduced vocabulary word embedding. The same vocabulary embedding will be used for the encoding RNN and the decoding RNN.

We generate a reduced vocabulary word embedding using the 40,000 or so most common words in the source content. We find the embedding weights of each vocabulary word that is handled by our chosen word embedding - we can use any word embedding such as word2vec, GloVe, sense2vec, ConceptNet-numberbatch etc. - and create a new matrix, literally copying the weights from the source word embedding.

We also want to include the embeddings of the words that are not part of the 40,000 most common words in the source content. We can do this by finding the word embedding that best matches these words. This is done by finding the vector of each word using the full word embedding and then finding the nearest embedding - within a given threshold - in the reduced vocabulary embedding.

### High-level how it works
