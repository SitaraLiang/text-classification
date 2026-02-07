import numpy as np
import matplotlib.pyplot as plt

import unicodedata
import codecs
import re
import os.path
import string
#import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.snowball import FrenchStemmer
from nltk.corpus import stopwords
#nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer

# Chargement des données speech
def load_pres(fname):
    alltxts = []
    alllabs = []
    s=codecs.open(fname, 'r','utf-8') # pour régler le codage
    while True:
        txt = s.readline()
        if(len(txt))<5:
            break
        #
        lab = re.sub(r"<[0-9]*:[0-9]*:(.)>.*","\\1",txt)
        txt = re.sub(r"<[0-9]*:[0-9]*:.>(.*)","\\1",txt)
        if lab.count('M') >0: # if a letter 'M' is present in the label (e.g., <100:12:M>), then we group the sentence to class -1
            alllabs.append(-1)
        else:
            alllabs.append(1) # else we group to class +1
        alltxts.append(txt)
    return alltxts,alllabs

def keep_only_part(txt_list, number):
    #A utiliser AVANT remove_ponctuation() sinon on peut pas reconnaitre les lignes

    if number is None:
        return txt_list
        
    tmp = txt_list
    for i in range(len(tmp)):
        tmp[i] = tmp[i].split('\n')
        if number > len(tmp[i]):
            print(f"error number{number} greater then length of text{len(tmp[i])}")
            return
        if number > 0:
            tmp[i] = '\n'.join(tmp[i][:number])
        else:
            tmp[i] = '\n'.join(tmp[i][number:])

    return tmp
    

def remove_caps(txt_list):

    tmp = txt_list
    for i in range(len(txt_list)):
        tmp[i].lower()

    return tmp

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

def stemming(txt_list):

    ps = PorterStemmer()
    tmp = txt_list
    for i in range(len(txt_list)):
        tmp[i] = ' '.join([ps.stem(word) for word in tmp[i].split()])
    #test = ps.stem("isnt")
    #print(test)
    return tmp

def stemming_french(txt_list):
    stemmer = FrenchStemmer()
    stemmed = []
    for txt in txt_list:
        words = txt.split()
        stemmed_words = [stemmer.stem(word) for word in words]
        stemmed.append(' '.join(stemmed_words))
    return stemmed
    
def change_capital_words(txt_list):

    tmp = txt_list
    for i in range(len(txt_list)):
        #if re.findall(r'\b[A-Z]+(?:\s+[A-Z]+)*\b', tmp[i]):
            #print("title found", re.findall(r'\b[A-Z]+(?:\s+[A-Z]+)*\b', tmp[i]))
        tmp[i] = re.sub(r'\b[A-Z]+(?:\s+[A-Z]+)*\b', 'TITLE',tmp[i])

    return tmp

def remove_numbers(txt_list):

    tmp = txt_list
    for i in range(len(tmp)):

        tmp[i] = re.sub('[0-9]+', '', tmp[i])

    return tmp

def vectorizer(txt_list, language):

    assert (language == 'FRENCH' or language == 'ENGLISH'), "Language value needs to be either FRENCH or ENGLISH"
    
    if language == "FRENCH":
        stop_list = stopwords.words('french')
    elif language == "ENGLISH":
        stop_list = stopwords.words('english')

    vectorizer = CountVectorizer(stop_words=stop_list)
    X = vectorizer.fit_transform(txt_list)

    return X, vectorizer

def remove_accents(text):
    """Remove accents from text"""
    # Parce que accent add noise
    # NFD = Normalization Form Decomposed, it tells Unicode to break characters into simpler pieces.
    # .decode("utf-8"): Converts the bytes back into a normal Python str.
    return unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')


def find_uninformative_words(alltxts, alllabs, threshold=0.8):
    """Find words that appear equally in both classes (not discriminative)"""
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Separate by class
    class_1_docs = [txt for txt, lab in zip(alltxts, alllabs) if lab == 1]
    class_2_docs = [txt for txt, lab in zip(alltxts, alllabs) if lab == -1]
    
    # Get word frequencies for each class
    vectorizer = CountVectorizer(lowercase=True, strip_accents='unicode')
    
    X1 = vectorizer.fit_transform(class_1_docs)
    vocab1 = vectorizer.get_feature_names_out()
    freq1 = np.asarray(X1.sum(axis=0)).flatten()
    
    X2 = vectorizer.fit_transform(class_2_docs)
    vocab2 = vectorizer.get_feature_names_out()
    freq2 = np.asarray(X2.sum(axis=0)).flatten()
    
    # Find words in both classes
    common_words = set(vocab1) & set(vocab2)
    
    # Calculate ratio of frequencies (close to 1 = balanced = uninformative)
    uninformative = []
    
    for word in common_words:
        idx1 = np.where(vocab1 == word)[0][0]
        idx2 = np.where(vocab2 == word)[0][0]
        
        f1 = freq1[idx1] / len(class_1_docs)
        f2 = freq2[idx2] / len(class_2_docs)
        
        ratio = min(f1, f2) / max(f1, f2)  # Close to 1 = balanced
        
        if ratio > threshold:  # Appears similarly in both classes
            uninformative.append((word, ratio, f1, f2))
    
    # Sort by how balanced they are
    uninformative.sort(key=lambda x: x[1], reverse=True)
    
    print("Top uninformative words (appear equally in both classes):")
    for word, ratio, f1, f2 in uninformative:
        print(f"{word:20s}: ratio={ratio:.3f}, class1={f1:.4f}, class2={f2:.4f}")
    
    return [w[0] for w in uninformative]
