import numpy as np
import matplotlib.pyplot as plt
import unicodedata
import codecs
import re
import os.path
import string
import sklearn
from collections import Counter
from wordcloud import WordCloud
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from wordcloud import STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer

def load_movies(path2data): # 1 classe par répertoire
    alltxts = [] # init vide
    labs = []
    cpt = 0
    stopwrds = stopwords.words('english')
    for cl in os.listdir(path2data): # parcours des fichiers d'un répertoire
        #print(cl)
        #print(type(cl))
        if cl == '.DS_Store':
            continue
        for f in os.listdir(path2data+cl):
            txt = open(path2data+cl+'/'+f).read()
            txt = txt.split(" ")
            txt = " ".join([word for word in txt if word not in stopwrds])
            alltxts.append(txt)
            labs.append(cpt)
        cpt+=1 # chg répertoire = cht classe

    return alltxts,labs

path = "data/movies1000/"
alltxts, alllabs = load_movies(path)

len_gen = sum([len(i.split(' ')) for i in alltxts])/len(alltxts)
poss_only = np.array(alltxts)[np.array(alllabs) == 0].tolist()
negs_only = np.array(alltxts)[np.array(alllabs) == 1].tolist()
print(f"Len all: {len(alltxts)} len pos: {len(poss_only)}, len neg: {len(negs_only)}")

len_pos = sum([len(i.split(' ')) for i in poss_only])/len(poss_only)
len_negs = sum([len(i.split(' ')) for i in negs_only])/len(negs_only)
print(f"Avergage general length: {len_gen}, negs length: {len_negs}, poss length: {len_pos}")

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

negs = ' '.join(np.array(alltxts)[np.array(alllabs) == 1].tolist())
poss = ' '.join(np.array(alltxts)[np.array(alllabs) == 0].tolist())

stop_words = set(STOPWORDS)
stop_words.update(["characters", "movies", "films", "way", "movie", "film", "even", "good", "much", "two", "first", "see", "scene", "story", "character", "one", "time", "make", "seem"])

wordcloud_neg = WordCloud(background_color='white', stopwords = stop_words, max_words=100).generate(negs)
wordcloud_neg.to_file('wordcloud_neg.png')
dict_neg = wordcloud_neg.process_text(negs)
tuple_neg = sorted(dict_neg.items(), key=lambda x: x[1], reverse=True)
print("top 10 neg:", tuple_neg[:15])

wordcloud_pos = WordCloud(background_color='white', stopwords = stop_words, max_words=100).generate(poss)
wordcloud_pos.to_file('wordcloud_pos.png')
dict_pos = wordcloud_pos.process_text(poss)
tuple_pos = sorted(dict_pos.items(), key=lambda x: x[1], reverse=True)
print("top 10 pos:", tuple_pos[:15])

corpus = [negs, poss]

vectorizer = TfidfVectorizer(max_features=100, stop_words=list(stop_words), ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names_out()
dense = tfidf_matrix.todense().tolist()

dict_tfidf_neg = dict(zip(feature_names, dense[0]))
dict_tfidf_pos = dict(zip(feature_names, dense[1]))

wordcloud_neg = WordCloud(background_color='white', max_words=100).generate_from_frequencies(dict_tfidf_neg)
wordcloud_neg.to_file('wordcloud_neg.png')
tuple_neg = sorted(dict_tfidf_neg.items(), key=lambda x: x[1], reverse=True)
print("top 15 neg (TF-IDF):", tuple_neg[:15])

wordcloud_pos = WordCloud(background_color='white', max_words=100).generate_from_frequencies(dict_tfidf_pos)
wordcloud_pos.to_file('wordcloud_pos.png')
tuple_pos = sorted(dict_tfidf_pos.items(), key=lambda x: x[1], reverse=True)
print("top 15 pos (TF-IDF):", tuple_pos[:15])
