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
#train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
#test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))
vn_gen = VnGen()
train_sents = vn_gen.gen_data(num_ex=100)
test_sents = vn_gen.gen_data(num_ex=20)
class NerCrf:
    def __init__(self,num_train,num_test):
        self.train_sents = vn_gen.gen_data(num_train)
        self.test_sents  = vn_gen.gen_data(num_test)
        self.crf = None
        self.label = []
    def word2features(self,sent, i):
        word = sent[i][0]
        postag = sent[i][1]
        
        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'postag': postag,
            'postag[:2]': postag[:2],        
        }
        if i > 0:
            word1 = sent[i-1][0]
            postag1 = sent[i-1][1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
            })
        else:
            features['BOS'] = True
            
        if i < len(sent)-1:
            word1 = sent[i+1][0]
            postag1 = sent[i+1][1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
            })
        else:
            features['EOS'] = True
                    
        return features


    def sent2features(self,sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    def sent2labels(self,sent):
        return [label for token, postag, label in sent]

    def sent2tokens(self,sent):
        return [token for token, postag, label in sent]
    def train(self):
        self.X_train = [self.sent2features(s) for s in self.train_sents]
        self.y_train = [self.sent2labels(s) for s in self.train_sents]

       # self.X_test = [self.sent2features(s) for s in self.test_sents]
       # self.y_test = [self.sent2labels(s) for s in self.test_sents]
        #print(X_test[1:3])
        self.crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs', 
            c1=0.1, 
            c2=0.1, 
            max_iterations=100, 
            all_possible_transitions=True
        )
        #print(len(X_train))
        self.crf.fit(self.X_train, self.y_train)
        self.labels = list(self.crf.classes_)
        self.labels.remove('O')
        self.save_model('./ner/crf_model.pkl')
    def test(self,string):
       # y_pred = crf.predict(X_test)
        test_s = []
        test_s.append(vn_gen.make_train_data(vn_gen.pos_tagging(string)))
        #  print("test_s",test_s)
        self.X_test = [self.sent2features(s) for s in test_s]
        self.y_test = [self.sent2labels(s) for s in test_s]
        y_pred = self.crf.predict(self.X_test)
        print("string:",string)
        print("y pred:", y_pred)
        print("y test:",self.y_test)
        s = ViPosTagger.postagging(ViTokenizer.tokenize(string))[0]
        print(s)
        return y_pred,self.y_test 
    def save_model(self,file_name):
        pk.dump(self,open(file_name,'wb'))
    def read_model(self,file_name):
        return pk.load(open(file_name,'rb'))
    def print_f1_score(self):
        a = 0
if __name__ == '__main__':
    k = NerCrf(2000,20)
    k.train()
    string = ' cho anh cổ  phiếu hcm giá 29 100 cổ  mua nhé'
    string = string.lower()
    print(string)
    k.test(string)
"""
    #print("x test",test_sents[1])
    #print("y_pred:",y_pred[1])
    metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)
    # group B and I results
    sorted_labels = sorted(  labels,  key=lambda name: (name[1:], name[0]))
    print(metrics.flat_classification_report( y_test, y_pred, labels=sorted_labels, digits=3))
    

    def print_transitions(trans_features):
        for (label_from, label_to), weight in trans_features:
            print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

    print("Top likely transitions:")
    print_transitions(Counter(crf.transition_features_).most_common(20))

    print("\nTop unlikely transitions:")
    print_transitions(Counter(crf.transition_features_).most_common()[-20:])
    def print_state_features(state_features):
        for (attr, label), weight in state_features:
            print("%0.6f %-8s %s" % (weight, label, attr))

    print("Top positive:")
    print_state_features(Counter(crf.state_features_).most_common(30))

    print("\nTop negative:")
    print_state_features(Counter(crf.state_features_).most_common()[-30:])
"""