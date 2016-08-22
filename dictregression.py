from csv import DictReader, DictWriter
from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
import time
from sklearn.preprocessing import LabelEncoder
# import random
import re
import sys
import wikipedia
import string
import nltk
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import numpy as np
from numpy import array
import omdb
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDRegressor
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import csv
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.ensemble import AdaBoostRegressor
from wiki2plain import Wiki2Plain

i =0
SENTENCE = 'question'
SOL = 'correctAnswer'
OPT1 = 'answerA'
OPT2 = 'answerB'
OPT3 = 'answerC'
OPT4 = 'answerD'
ID = 'id'


oh = DictWriter(open("wiki2_1.csv", 'w'), ["id", "Wiki"])
oh.writeheader()
            
train = list(DictReader(open("sci_train.csv", 'r')))
test = list(DictReader(open("sci_test.csv", 'r')))
            
# Tokenizer for CountVectorizer for stemming using Wordnet Corpora
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        nopunctutation = "".join(
            [ch for ch in doc if ch not in string.punctuation])
        lemma = [self.wnl.lemmatize(t, pos="v")
                 for t in nltk.word_tokenize(nopunctutation)]
         
        return lemma


def bigrams_per_line(doc):
    for ln in doc.split('\n'):
        terms = re.findall(r'\w{2,}', ln)
        for bigram in zip(terms, terms[1:]):
            yield '%s %s' % bigram


def rpunctuation(text):
    regex = re.compile('[%s|0-9]' % re.escape(string.punctuation))
    return regex.sub('', text)

def rightoptionkeyword(ids,solution):
    #print ids
    #wiki = list(DictReader(open("../../data/wikiAnswers2.csv", 'r')))
    for X in train:
        if(X['id'] == ids):
            if solution == 'A':
                answer = X[OPT1]
            if solution == 'B':
                answer = X[OPT2]
            if solution == 'C':
                answer = X[OPT3]
            if solution == 'D':
                answer = X[OPT4]
            return answer
        else:
            return '\0'

def rightoptionkeywordwikidata(ids):
    #print ids
    wiki = list(DictReader(open("wikianwserfortrain.csv", 'r')))
    for X in wiki:
        if(X['id'] == ids):
            return X['Wiki']
        else:
            return '\0'

def wikidataforsentence(ids):
    #print ids
    wiki = list(DictReader(open("wikitrain.csv", 'r')))
    for X in wiki:
        if(X['id'] == ids):
            return X['Wiki']
        else:
            return '\0'


def wikidataforsentenceTest(ids):
    #print ids
    wiki = list(DictReader(open("wikiTest.csv", 'r')))
    for X in wiki:
        if(X['id'] == ids):
            return X['Wiki']
        else:
            return '\0'
class Featurizer:
    def __init__(self):
        # use: uni-grams, bi-grams, stopwords
        # punctation removal
        self.vectorizer = CountVectorizer(
            stop_words = 'english',
            #strip_accents='unicode',	
            lowercase= True ,
            analyzer='word',
            #max_df = 1,
            #preprocessor=rpunctuation,
            ngram_range=(1,1),
            min_df = 1)

    def train_feature(self, examples):
        print "In train"
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        print "In test"
        return self.vectorizer.transform(examples)

    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        if len(categories) == 4:
            top10 = np.argsort(classifier.coef_[0])[-10:]
            bottom10 = np.argsort(classifier.coef_[0])[:10]
            #print("Pos: %s" % "\n ".join(feature_names[top10]))
            print("\n\n****************************************\n\n")
            #print("Neg: %s" % " ".join(feature_importance_[top10]))
        else:
            for i, category in enumerate(categories):
                top10 = np.argsort(classifier.coef_[i])[-10:]
                print("%s: %s" % (category, " ".join(feature_names[top10])))


class classify:
    
    
    def __init__(self):
        self.x_train = []
        self.x_test = []
        self.y = []

        self.test = []
        self.train = []        

        self.quesFeat = {}        
        self.predictions = []
        
        self.cv_train = []
        self.cv_test = []
        self.cv_y_train = []
        self.cv_y_test = []

    def questionFeatures(self):
        wiki = list(DictReader(open("wikiTest.csv", 'r')))
        for X in wiki:
            if(X['id'] == ids):
                return X['Wiki']
        else:
            return '\0'
        questions = list(DictReader(open("questions.csv", 'rU')))
        for ques in questions:
            unigrams = {}
            unigrams['_length_'] = len(ques['questionText'])
            unigrams['_cat_'] = ques['cat']
            unigrams['_answer_'] = ques['answer']
            self.quesFeat[ques['question']] = unigrams     


if __name__ == "__main__":

    # Cast to list to keep it all in memory
    
    #~ random.shuffle(train)

    #~ test = train[:int(len(train) * .2)]
    #~ train = train[int(len(train) * .8):]

    feat = Featurizer()



    labels = []
    for line in train:
        if line[SOL] == 'A':
            answer = line[OPT1]
        if line[SOL] == 'B':
            answer = line[OPT2]
        if line[SOL] == 'C':
            answer = line[OPT3]
        if line[SOL] == 'D':
            answer = line[OPT4]
            
        if not answer in labels:
            labels.append(answer)

    print("Label set: %s" % str(labels))
    lb = LabelEncoder()
    y = lb.fit_transform(labels)
    print y
    x_train = feat.train_feature(
        x[SENTENCE] + ' ' + rightoptionkeyword(x[ID],x[SOL]) + ' ' + rightoptionkeywordwikidata(x[ID])
        for x in train)
    #print x_train;
    #' ' + wikidataforsentence(x[ID])+
    x_test = feat.test_feature( 
        x[SENTENCE] + ' '+ wikidataforsentenceTest(x[ID]) 
        for x in test)

    #y_train = array(list(labels.index(x[SOL])
    #                     for x in train))

    #y_test = array(list(labels.index(x[SOL])
    #                     for x in test))

    #print(len(train), len(y_train))
    #print(set(y_train))
    
    # Train classifier
    #lr = SGDClassifier(loss='log', penalty='l2', shuffle=True) #learning_rate = 'optimal'
    #SGDClassifier(loss='log', penalty='l1' , shuffle=True)
    print "In Predict"
    clf = linear_model.Lasso(alpha=0.01, fit_intercept=True)
    print "1"
    clf = clf.fit(x_train, y)
    print "2"
    predictions = clf.predict(x_test)
    print "done"
    print predictions
    newpredict = lb.inverse_transform(y)  
    print "still there"
    print newpredict
    print "In writePredictions"


    #lr.fit(x_train, y_train)
    #feat.show_top10(lr, labels)
    #predictions = lr.predict(x_test)
    #   print "Logs " ,len(predictions), " test ", len(y_test)
    #print accuracy_score(y_test,predictions)


    o = DictWriter(open("predictionscheck.csv", 'w'), ["id", "correctAnswer"])
    o.writeheader()
    for ii, pp in zip([x['id'] for x in test], newpredict):
        d = {'id': ii, 'correctAnswer': pp}
        o.writerow(d)
