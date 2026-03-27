import numpy as np
import matplotlib.pyplot as plt
#from utils import *
import unicodedata
import codecs
import re
import os.path
import string
#import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
#nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#from wordcloud import WordCloud 
from wordcloud import STOPWORDS

def load_movies(path2data): # 1 classe par répertoire
    alltxts = [] # init vide
    labs = []
    cpt = 0
    for cl in os.listdir(path2data): # parcours des fichiers d'un répertoire
        #print(cl)
        #print(type(cl))
        if cl == '.DS_Store':
            continue
        for f in os.listdir(path2data+cl):
            txt = open(path2data+cl+'/'+f).read()
            alltxts.append(txt)
            labs.append(cpt)
        cpt+=1 # chg répertoire = cht classe

    return alltxts,labs

path = "data/movies1000/"
alltxts, alllabs = load_movies(path)

def remove_ponctuation(txt_list):

    tmp = txt_list
    punc = string.punctuation
    #print(punc)
    punc += '\n\r\t'
    for i in range(len(txt_list)):
        #tmp[i] = re.sub(r"\b's\b", '', tmp[i])
        #tmp[i] = tmp[i].translate(str.maketrans(punc, ' ' * len(punc)))
        tmp[i] = tmp[i].translate(str.maketrans('', '', punc))

    return tmp

def remove_caps(txt_list):

    tmp = txt_list
    for i in range(len(txt_list)):
        tmp[i].lower()

    return tmp

alltxts = remove_caps(alltxts)
alltxts = remove_ponctuation(alltxts)

stop_words = set(STOPWORDS)
stop_words.update(["characters", "movies", "films", "way", "movie", "film", "even", "good", "much", "two", "first", "see", "scene", "story", "character", "one", "time", "make", "seem"])
tokenizer = TfidfVectorizer().build_tokenizer()
stop_words = sum([tokenizer(sw) for sw in stop_words], [])

vectorizer = TfidfVectorizer(lowercase=True, stop_words=list(stop_words))
X = vectorizer.fit_transform(alltxts)
vocab = vectorizer.get_feature_names_out()

x_train, x_test, y_train, y_test = train_test_split(X, alllabs, test_size = 0.2, random_state=10, shuffle=True)

#Naïve Bayes
nb_clf = MultinomialNB()
nb_clf.fit(x_train, y_train)


#Logistic Regression
t = 1e-8
C=100.0
lr_clf = LogisticRegression(random_state=0, solver='liblinear',max_iter=100, tol=t, C=C)
lr_clf.fit(x_train, y_train)

#Linear SVM
svm_clf = LinearSVC(random_state=0)
svm_clf.fit(x_train, y_train)

pred_nbt = nb_clf.predict(x_train)
pred_lrt = lr_clf.predict(x_train)
pred_svmt = svm_clf.predict(x_train)

pred_nb = nb_clf.predict(x_test)
pred_lr = lr_clf.predict(x_test)
pred_svm = svm_clf.predict(x_test)


print(f"Naive Bayes accuracy train={accuracy_score(y_train, pred_nbt)}, accuracy test={accuracy_score(y_test, pred_nb)}")
print(f"Logistic Regression accuracy train={accuracy_score(y_train, pred_lrt)}, accuracy test={accuracy_score(y_test, pred_lr)}")
print(f"SVM accurac ytrain={accuracy_score(y_train, pred_svmt)}, accuracy test={accuracy_score(y_test, pred_svm)}")

print(f"Naive Bayes precision train={precision_score(y_train, pred_nbt)}, precision test={precision_score(y_test, pred_nb)}")
print(f"Logistic Regression precision train={precision_score(y_train, pred_lrt)}, precision test={precision_score(y_test, pred_lr)}")
print(f"SVM precision train={precision_score(y_train, pred_svmt)}, precision test={precision_score(y_test, pred_svm)}")

print(f"Naive Bayes recall train={recall_score(y_train, pred_nbt)}, recall test={recall_score(y_test, pred_nb)}")
print(f"Logistic Regression recall train={recall_score(y_train, pred_lrt)}, recall test={recall_score(y_test, pred_lr)}")
print(f"SVM recall train={recall_score(y_train, pred_svmt)}, recall test={recall_score(y_test, pred_svm)}")

print(f"Naive Bayes F1 train={f1_score(y_train, pred_nbt)}, F1 test={f1_score(y_test, pred_nb)}")
print(f"Logistic Regression F1 train={f1_score(y_train, pred_lrt)}, F1 test={f1_score(y_test, pred_lr)}")
print(f"SVM F1 train={f1_score(y_train, pred_svmt)}, F1 test={f1_score(y_test, pred_svm)}")
