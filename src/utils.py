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
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score


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
        if lab.count('M') >0: # if a letter 'M' is present in the label (e.g., <100:12:M>), then we group the sentence to class 1
            alllabs.append(1)
        else:
            alllabs.append(0) # else we group to class 0
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


def find_uninformative_words(alltxts, alllabs, threshold=0.85, global_max_freq=0.01):
    """
    Finds words that are either:
    1. Too balanced between classes (uninformative for classification)
    2. Too frequent globally (likely functional "glue" words like 'plus', 'tout')
    """
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer

    # 1. Separate classes
    class_1_docs = [txt for txt, lab in zip(alltxts, alllabs) if lab == 1]
    class_0_docs = [txt for txt, lab in zip(alltxts, alllabs) if lab == 0]
    
    # 2. Efficient frequency helper
    def get_freqs_dict(docs):
        vec = CountVectorizer(lowercase=True, strip_accents='unicode')
        X = vec.fit_transform(docs)
        counts = np.asarray(X.sum(axis=0)).flatten()
        return {word: count / len(docs) for word, count in zip(vec.get_feature_names_out(), counts)}

    freqs1 = get_freqs_dict(class_1_docs)
    freqs0 = get_freqs_dict(class_0_docs)
    
    # 3. Global frequency check (The Zipf Pruner)
    full_vec = CountVectorizer(lowercase=True, strip_accents='unicode')
    X_full = full_vec.fit_transform(alltxts)
    global_map = dict(zip(full_vec.get_feature_names_out(), 
                          np.asarray(X_full.sum(axis=0)).flatten() / len(alltxts)))

    common_words = set(freqs1.keys()) & set(freqs0.keys())
    uninformative = []
    
    for word in common_words:
        f1 = freqs1[word]
        f0 = freqs0[word]
        ratio = min(f1, f0) / max(f1, f0)
        
        # LOGIC: 
        # Is it too balanced? (ratio > 0.85)
        # OR is it a "Top 1%" heavy-lifter? (global_freq > 0.01)
        if ratio > threshold or global_map.get(word, 0) > global_max_freq:
            uninformative.append(word)
            
    print(f"Filtering {len(uninformative)} uninformative/heavy words...")
    return uninformative


def evaluate_model(name, model, X_test, y_test):
    # 1. Get Hard Predictions for F1
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds)

    # 2. Get Scores/Probs for AUC and AP
    if hasattr(model, "predict_proba"):
        # For NB and Logistic Regression
        probs = model.predict_proba(X_test)[:, 1]
    else:
        # For LinearSVC (uses distance from hyperplane)
        probs = model.decision_function(X_test)

    auc = roc_auc_score(y_test, probs)
    ap = average_precision_score(y_test, probs)

    print(f"--- {name} ---")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")
    print(f"Avg Prec: {ap:.4f}\n")
