pyenv install 3.6.6
pyenv rehash
pyenv virtualenv 3.6.6 summarisers
pyenv local summarisers

pip install -U pip

pip install -r requirements.txt

python -m spacy download en