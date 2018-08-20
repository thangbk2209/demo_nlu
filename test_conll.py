import matplotlib.pyplot as plt
plt.style.use('ggplot')
from itertools import chain

import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from vn_gen import VnGen
from collections import Counter
from pyvi import ViPosTagger,ViTokenizer
import pickle as pk
nltk.corpus.conll2002.fileids()
#%%time
train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))
print("train_sents",train_sents[:3])