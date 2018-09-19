# summarisers

messing with some approaches to summarising text with machine learning

## pre-requisites

- Python 3.6.6 / pip 18.0 (I use Pyenv and virutalenv)
- Tensorflow

### Preparing the environment

The `setup.sh` script may be helpful in setting up your environment, assuming you have already installed `pyenv` and `virtualenv` (see my tutorial [Python dependency - hell no!](http://www.webpusher.ie/2018/09/19/python-dependency-hell-no/))

The script contains the following

```bash
pyenv install 3.6.6
pyenv rehash
pyenv virtualenv 3.6.6 summarisers
pyenv local summarisers

pip intall -U pip

pip install -r requirements.txt
```

Once you run that you should be ready to go.

## Example 1 - Sequence to Sequence

LSTM with Attention

## Example 2 - BiLSTM with Attention

## Example 3 - Seq2Seq with pointer and coverage
